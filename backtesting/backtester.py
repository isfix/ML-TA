# backtesting/backtester.py
"""
Core backtesting engine for simulating trading strategies.
Implements Option A2: Incremental data fetching from MT5 (simulated live)
within the backtest loop, dynamic feature generation, ATR Stop Loss,
dynamic Take Profit, flexible position sizing, and breakeven.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone

# Assuming config, logging_utils, file_utils, and other necessary modules are accessible
try:
    import config
    from utilities import logging_utils, file_utils
    # FeatureBuilder will be instantiated per cycle or per pair if its state is light
    from feature_engineering.feature_builder import FeatureBuilder
    # DataFetcher, MarketStructureAnalyzer, RiskManager passed in __init__
    # SignalProcessor created via factory function
except ImportError:
    print("FATAL: Could not perform standard imports in Backtester. Ensure paths are correct.")
    raise

logger = logging_utils.setup_logger(__name__, config.LOG_FILE_BACKTEST) # Use dedicated backtest log


class Backtester:
    def __init__(self, config_obj, logger_obj, data_fetcher, market_analyzer,
                 model_loader_func, signal_processor_creator_func, risk_manager):
        self.config = config_obj
        self.logger = logger_obj # Use the passed logger instance

        self.data_fetcher = data_fetcher
        self.market_analyzer = market_analyzer
        self.model_loader_func = model_loader_func # func(pair_key) -> (pipeline, feature_names)
        self.signal_processor_creator_func = signal_processor_creator_func # func(pipeline, f_names) -> SignalProcessor
        self.risk_manager = risk_manager

        self.initial_capital = self.config.BACKTEST_INITIAL_CAPITAL
        self.current_equity = self.initial_capital
        self.spread_pips = self.config.BACKTEST_SPREAD_PIPS

        # SL/TP/Breakeven params from config
        self.use_breakeven = getattr(self.config, 'USE_BREAKEVEN', True)
        self.breakeven_trigger_pips = getattr(self.config, 'BREAKEVEN_TRIGGER_PIPS', 20)
        self.breakeven_adjust_pips = getattr(self.config, 'BREAKEVEN_ADJUST_PIPS', 2)
        # Trailing stop params (can be added later if USE_TRAILING_STOP is True)

        self.backtest_pairs_keys = self.config.PRIMARY_MODEL_PAIRS_TIMEFRAMES
        self.pair_configs = {} # Store pip_size, point_value_per_lot, etc.
        for key in self.backtest_pairs_keys:
            cfg_data = self.config.HISTORICAL_DATA_SOURCES.get(key)
            if cfg_data:
                self.pair_configs[key] = {
                    "pair_name": cfg_data['pair'],
                    "pip_size": cfg_data['pip_size'],
                    # For dynamic risk sizing in backtest:
                    "backtest_mt5_point": cfg_data.get("backtest_mt5_point"),
                    "backtest_mt5_trade_tick_value": cfg_data.get("backtest_mt5_trade_tick_value"),
                    "backtest_mt5_volume_min": cfg_data.get("backtest_mt5_volume_min"),
                    "backtest_mt5_volume_max": cfg_data.get("backtest_mt5_volume_max"),
                    "backtest_mt5_volume_step": cfg_data.get("backtest_mt5_volume_step"),
                }
            else:
                self.logger.error(f"Configuration for M5 pair key '{key}' not found in HISTORICAL_DATA_SOURCES. It will be skipped.")
        # Filter out pairs without config
        self.backtest_pairs_keys = [k for k in self.backtest_pairs_keys if k in self.pair_configs]


        self.trade_log = []
        self.equity_curve_data = []
        self.active_trades = {} # {m5_pair_key: trade_dict}

        # Store loaded models and signal processors to avoid reloading every candle
        self.loaded_models_cache = {} # {m5_pair_key: (pipeline, feature_names)}
        self.signal_processors_cache = {} # {m5_pair_key: SignalProcessor_instance}

        self.logger.info(f"Backtester initialized for Option A2. Capital: ${self.initial_capital:,.2f}, Spread: {self.spread_pips} pips")
        self.logger.info(f"Backtesting M5 primary pairs: {self.backtest_pairs_keys}")

    def _get_pair_config_details(self, m5_pair_key):
        return self.pair_configs.get(m5_pair_key)

    def _apply_spread(self, price, trade_type, pip_size):
        spread_amount = self.spread_pips * pip_size
        return price + spread_amount if trade_type == 'BUY' else price - spread_amount

    def _calculate_pnl(self, entry_price, exit_price, trade_type, lot_size, pip_size, pair_key):
        pair_cfg = self._get_pair_config_details(pair_key)
        if not pair_cfg or not pair_cfg.get("backtest_mt5_point") or not pair_cfg.get("backtest_mt5_trade_tick_value"):
            self.logger.error(f"Missing point/tick_value for P&L calc for {pair_key}. Returning 0 PNL.")
            return 0.0
        
        point_size = pair_cfg["backtest_mt5_point"]
        tick_value_per_lot = pair_cfg["backtest_mt5_trade_tick_value"] # Value of 1 point move for 1 lot

        price_diff_points = (exit_price - entry_price) if trade_type == 'BUY' else (entry_price - exit_price)
        price_diff_in_min_increments = price_diff_points / point_size
        
        pnl = price_diff_in_min_increments * tick_value_per_lot * lot_size
        return pnl

    def run_backtest(self):
        self.logger.info("Starting backtest run (Option A2: Incremental MT5 Data Simulation)...")
        
        try:
            start_dt_config = datetime.fromisoformat(self.config.START_DATE_HISTORICAL).replace(tzinfo=timezone.utc)
            end_dt_config = datetime.fromisoformat(self.config.END_DATE_HISTORICAL).replace(tzinfo=timezone.utc)
        except ValueError as e:
            self.logger.error(f"Invalid date format in config: {e}. Aborting backtest."); return None, None, None

        # Create M5 simulation timeline
        simulation_timestamps = pd.date_range(start=start_dt_config, end=end_dt_config, freq='5min', tz='UTC')
        if simulation_timestamps.empty:
            self.logger.error("Simulation timestamp range is empty. Check start/end dates."); return None, None, None
        
        self.equity_curve_data.append((simulation_timestamps[0] - pd.Timedelta(minutes=5), self.initial_capital)) # Equity before first bar

        # Instantiate FeatureBuilder once
        # It needs MarketStructureAnalyzer, which is already an instance variable
        feature_builder_instance = FeatureBuilder(self.config, self.logger, self.market_analyzer)

        for current_sim_m5_time in simulation_timestamps:
            self.logger.debug(f"--- Processing Timestamp: {current_sim_m5_time} ---")
            self.equity_curve_data.append((current_sim_m5_time, self.current_equity))

            for m5_pair_key in self.backtest_pairs_keys:
                pair_cfg_details = self._get_pair_config_details(m5_pair_key)
                if not pair_cfg_details: continue # Should have been filtered in __init__

                pair_name = pair_cfg_details['pair_name']
                pip_size = pair_cfg_details['pip_size']
                
                # --- 1. Simulated Data Fetching for current_sim_m5_time ---
                # H1 data: fetch window ending around the H1 bar relevant to current_sim_m5_time
                h1_timeframe_enum = self.config.HISTORICAL_DATA_SOURCES[f"{pair_name}_{self.config.TIMEFRAME_H1_STR}"]['mt5_timeframe']
                h1_end_dt = current_sim_m5_time.replace(minute=0, second=0, microsecond=0) # Current H1 bar start
                h1_start_dt = h1_end_dt - timedelta(hours=self.config.H1_SNR_DATA_FETCH_WINDOW -1)
                df_h1_segment = self.data_fetcher.fetch_historical_data(pair_name, h1_timeframe_enum, h1_start_dt, h1_end_dt)

                # M5 data: fetch window ending at current_sim_m5_time
                m5_timeframe_enum = self.config.HISTORICAL_DATA_SOURCES[m5_pair_key]['mt5_timeframe']
                m5_end_dt = current_sim_m5_time
                m5_start_dt = m5_end_dt - timedelta(minutes=(self.config.LIVE_NUM_CANDLES_FETCH_M5 -1) * 5) # Approx
                df_m5_segment = self.data_fetcher.fetch_historical_data(pair_name, m5_timeframe_enum, m5_start_dt, m5_end_dt)

                if df_m5_segment is None or df_m5_segment.empty:
                    self.logger.debug(f"No M5 data for {pair_name} at {current_sim_m5_time}. Skipping pair cycle.")
                    continue
                # H1 data is optional if no H1 features are configured to be critical
                if (df_h1_segment is None or df_h1_segment.empty) and self.config.H1_NUM_SNR_LEVELS > 0:
                    self.logger.debug(f"No H1 data for {pair_name} at {current_sim_m5_time} for SNR. H1 features will be NaN.")
                    # Continue with df_h1_segment as None, FeatureBuilder should handle it

                # --- 2. Dynamic Feature Engineering for the latest M5 candle in the segment ---
                # build_features_for_live should process the segments and return features for all,
                # then we take the last one.
                all_features_df = feature_builder_instance.build_features_for_live(df_m5_segment, df_h1_segment)
                if all_features_df.empty or current_sim_m5_time not in all_features_df.index:
                    self.logger.debug(f"Feature generation failed or no features for {current_sim_m5_time} for {pair_name}.")
                    continue
                
                latest_candle_features_series = all_features_df.loc[current_sim_m5_time]
                # This series now contains M5 OHLCV, M5 indicators, H1 dynamic SNR, H1 indicators, etc.

                # --- 3. Load Model & Signal Processor (if not cached) ---
                if m5_pair_key not in self.loaded_models_cache:
                    pipeline, f_names = self.model_loader_func(m5_pair_key)
                    if pipeline and f_names:
                        self.loaded_models_cache[m5_pair_key] = (pipeline, f_names)
                        self.signal_processors_cache[m5_pair_key] = self.signal_processor_creator_func(pipeline, None, f_names) # Scaler is in pipeline
                        self.logger.info(f"Model and SignalProcessor loaded for {pair_name} for backtest.")
                    else:
                        self.logger.error(f"Failed to load model for {pair_name}. Skipping pair for this cycle."); continue
                
                signal_processor = self.signal_processors_cache[m5_pair_key]

                # --- 4. Trade Management & Signal Generation ---
                # Current M5 bar for SL/TP checks is the one AT current_sim_m5_time
                # Entry signal is based on features of candle ending AT current_sim_m5_time (latest_candle_features_series)
                # Entry happens at OPEN of NEXT bar (current_sim_m5_time + 5 mins) - this needs careful handling of indices.
                # For simplicity, let's assume entry on open of current_sim_m5_time based on PREVIOUS bar's signal.
                # So, latest_candle_features_series is for candle T-1, decision for candle T.
                # The `all_features_df` should contain features for `current_sim_m5_time`.
                # The signal for `current_sim_m5_time`'s open is based on `latest_candle_features_series` which should be for the *previous* M5 close.
                # Let's adjust: features are for candle ending at `current_sim_m5_time - 5min`.
                
                # To get features for the candle that JUST CLOSED before current_sim_m5_time's open:
                prev_m5_time = current_sim_m5_time - pd.Timedelta(minutes=5)
                if prev_m5_time not in all_features_df.index:
                    self.logger.debug(f"No features for previous candle {prev_m5_time} for {pair_name}.")
                    # This means we can't make a decision for current_sim_m5_time's open.
                    # We still need to manage open trades using current_sim_m5_time's OHLC.
                    # For now, let's assume all_features_df.loc[current_sim_m5_time] are features of the candle that just closed.
                    # And decisions are made based on that for the *next* candle.
                    # This means the loop should be structured so that `current_sim_m5_time` is the *open* of the bar we act on.
                    # The features used are from `current_sim_m5_time - 5min`.

                    # Let's simplify: `latest_candle_features_series` are features of candle `T`.
                    # Trade management uses OHLC of candle `T`.
                    # New entry signal from candle `T` features applies to open of `T+1`.
                    # This requires a slight re-think of the loop or how `current_sim_m5_time` is used.

                    # For now: `current_m5_bar_for_sltp_checks` is `df_m5_segment.loc[current_sim_m5_time]`
                    # `signal_features_series` is `all_features_df.loc[current_sim_m5_time]`
                    # Entry price for a new trade will be `df_m5_segment.loc[current_sim_m5_time, self.config.OPEN_COL]` (if signal is from T-1)
                    # OR, if signal from T, entry at T's close / T+1 open.
                    # P1's live trading used features of latest complete candle for signal, then market order.
                    # Let's assume `latest_candle_features_series` are for the candle that just closed at `current_sim_m5_time`.
                    # Trade management happens on this candle's H/L. New entries are considered at its close for next bar's open.

                current_m5_ohlc_bar = df_m5_segment.loc[current_sim_m5_time] # OHLC of the current bar for SL/TP checks

                # Manage open trade for this pair
                trade_closed_this_bar = False
                if m5_pair_key in self.active_trades:
                    trade = self.active_trades[m5_pair_key]
                    exit_price, reason = None, None
                    
                    # Breakeven Check
                    if self.use_breakeven and not trade.get('is_breakeven', False):
                        profit_pips_be = 0
                        if trade['type'] == 'BUY': profit_pips_be = (current_m5_ohlc_bar[self.config.HIGH_COL] - trade['entry_price_adj']) / pip_size
                        else: profit_pips_be = (trade['entry_price_adj'] - current_m5_ohlc_bar[self.config.LOW_COL]) / pip_size
                        
                        if profit_pips_be >= self.breakeven_trigger_pips:
                            new_sl_be = trade['entry_price_adj'] + (self.breakeven_adjust_pips * pip_size if trade['type'] == 'BUY' else -self.breakeven_adjust_pips * pip_size)
                            if (trade['type'] == 'BUY' and new_sl_be > trade['sl_price']) or \
                               (trade['type'] == 'SELL' and new_sl_be < trade['sl_price']): # Ensure SL is improved
                                trade['sl_price'] = new_sl_be
                                trade['is_breakeven'] = True
                                self.logger.info(f"[{current_sim_m5_time}] {pair_name} {trade['type']} Breakeven set. New SL: {new_sl_be:.5f}")
                    
                    # SL/TP Check
                    if trade['type'] == 'BUY':
                        if current_m5_ohlc_bar[self.config.LOW_COL] <= trade['sl_price']: exit_price, reason = trade['sl_price'], "Stop Loss"
                        elif trade.get('tp_price') and current_m5_ohlc_bar[self.config.HIGH_COL] >= trade['tp_price']: exit_price, reason = trade['tp_price'], "Take Profit"
                    else: # SELL
                        if current_m5_ohlc_bar[self.config.HIGH_COL] >= trade['sl_price']: exit_price, reason = trade['sl_price'], "Stop Loss"
                        elif trade.get('tp_price') and current_m5_ohlc_bar[self.config.LOW_COL] <= trade['tp_price']: exit_price, reason = trade['tp_price'], "Take Profit"
                    
                    # Fallback EMA Exit (if no TP was set initially)
                    if exit_price is None and not trade.get('tp_price'):
                        m5_exit_ema_val = latest_candle_features_series.get(self.config.M5_EXIT_EMA_COL)
                        if pd.notna(m5_exit_ema_val):
                            if trade['type'] == 'BUY' and current_m5_ohlc_bar[self.config.CLOSE_COL] < m5_exit_ema_val:
                                exit_price, reason = current_m5_ohlc_bar[self.config.CLOSE_COL], f"M5 EMA Exit ({self.config.M5_EXIT_EMA_PERIOD})"
                            elif trade['type'] == 'SELL' and current_m5_ohlc_bar[self.config.CLOSE_COL] > m5_exit_ema_val:
                                exit_price, reason = current_m5_ohlc_bar[self.config.CLOSE_COL], f"M5 EMA Exit ({self.config.M5_EXIT_EMA_PERIOD})"

                    if exit_price is not None:
                        pnl = self._calculate_pnl(trade['entry_price_adj'], exit_price, trade['type'], trade['volume'], pip_size, m5_pair_key)
                        self.current_equity += pnl
                        self.trade_log.append({
                            "Pair": pair_name, "Timestamp_Open": trade['entry_time'], "Type": trade['type'],
                            "Entry_Price": trade['entry_price_adj'], "SL_Price": trade['original_sl_price'], "TP_Price": trade.get('tp_price'),
                            "Volume": trade['volume'], "Timestamp_Close": current_sim_m5_time, "Exit_Price": exit_price, 
                            "Reason": reason, "P&L": pnl, "Equity_After_Trade": self.current_equity
                        })
                        self.logger.info(f"[{current_sim_m5_time}] {pair_name} {trade['type']} CLOSED by {reason} @ {exit_price:.5f}. P&L: {pnl:.2f}. Equity: {self.current_equity:.2f}")
                        del self.active_trades[m5_pair_key]; trade_closed_this_bar = True

                # Check for new entry signal if no active trade for this pair
                if not trade_closed_this_bar and m5_pair_key not in self.active_trades:
                    candidate_direction = signal_processor.check_candidate_entry_conditions(latest_candle_features_series)
                    if candidate_direction:
                        ml_confirmed_signal = signal_processor.generate_ml_confirmed_signal(latest_candle_features_series, candidate_direction)
                        if ml_confirmed_signal:
                            signal_type = ml_confirmed_signal['signal'] # 'BUY' or 'SELL'
                            self.logger.info(f"[{current_sim_m5_time}] {pair_name} New {signal_type} signal. Prob: {ml_confirmed_signal['probability']:.3f}")

                            # Entry at open of *next* bar. For simulation, use current bar's close as proxy or next open if available.
                            # Let's assume entry at current bar's close for simplicity in backtest.
                            entry_price_raw = current_m5_ohlc_bar[self.config.CLOSE_COL]
                            entry_price_adj = self._apply_spread(entry_price_raw, signal_type, pip_size)
                            
                            atr_val_for_sl = latest_candle_features_series.get(self.config.M5_ATR_COL_BASE)
                            if pd.isna(atr_val_for_sl) or atr_val_for_sl <= 0:
                                self.logger.warning(f"Invalid ATR for SL calc ({atr_val_for_sl}) for {pair_name}. Skipping trade."); continue

                            sl_price = self.risk_manager.calculate_stop_loss(entry_price_adj, atr_val_for_sl, signal_type, pair_name)
                            tp_price = self.risk_manager.calculate_take_profit(entry_price_adj, sl_price, signal_type, pair_name)
                            # TODO: Dynamic TP from FeatureBuilder/MarketStructure (H1/M5 pivots) could override RRR based TP.
                            # This needs `latest_candle_features_series` to have these pivot levels.
                            # For now, RiskManager's RRR-based TP is used.

                            if sl_price is None: self.logger.warning(f"SL calculation failed for {pair_name}. Skipping trade."); continue
                            # Basic validation: SL must give some room, TP (if set) must be beyond entry + spread
                            if (signal_type == 'BUY' and (sl_price >= entry_price_adj or (tp_price and tp_price <= entry_price_adj))) or \
                               (signal_type == 'SELL' and (sl_price <= entry_price_adj or (tp_price and tp_price >= entry_price_adj))):
                                self.logger.warning(f"Invalid SL/TP for {pair_name} {signal_type}: E={entry_price_adj:.5f} SL={sl_price:.5f} TP={tp_price}. Skipping.")
                                continue

                            trade_volume = self.risk_manager.get_trade_volume(
                                account_balance=self.current_equity,
                                entry_price=entry_price_adj, sl_price=sl_price,
                                symbol=pair_name, symbol_info_mt5=None # None for backtest, uses config
                            )
                            if trade_volume <= 0: self.logger.error(f"Calculated trade volume is {trade_volume} for {pair_name}. Skipping trade."); continue

                            self.active_trades[m5_pair_key] = {
                                'type': signal_type, 'entry_time': current_sim_m5_time, # Or next bar's open time
                                'entry_price_raw': entry_price_raw, 'entry_price_adj': entry_price_adj,
                                'sl_price': sl_price, 'original_sl_price': sl_price, 'tp_price': tp_price,
                                'volume': trade_volume, 'is_breakeven': False, 'pair_name': pair_name
                            }
                            self.logger.info(f"[{current_sim_m5_time}] {pair_name} OPENED {signal_type} @ {entry_price_adj:.5f} (Raw: {entry_price_raw:.5f}), Vol: {trade_volume:.2f}, SL: {sl_price:.5f}, TP: {tp_price if tp_price else 'Dynamic/EMA'}")
            
            # End of pair loop
        # End of timestamp loop

        self.logger.info("End of backtest simulation period. Closing any remaining open trades...")
        for m5_pair_key_open, trade_open in list(self.active_trades.items()): # Use list for safe deletion
            pair_cfg_open = self._get_pair_config_details(m5_pair_key_open)
            # Fetch very last known M5 bar for this pair to close
            last_m5_data = self.data_fetcher.fetch_historical_data(pair_cfg_open['pair_name'], 
                                                                  self.config.HISTORICAL_DATA_SOURCES[m5_pair_key_open]['mt5_timeframe'],
                                                                  end_dt_config - timedelta(minutes=5), # Get a small window
                                                                  end_dt_config)
            if last_m5_data is not None and not last_m5_data.empty:
                last_close_price = last_m5_data[self.config.CLOSE_COL].iloc[-1]
                last_timestamp = last_m5_data.index[-1]
                pnl_eod = self._calculate_pnl(trade_open['entry_price_adj'], last_close_price, trade_open['type'], trade_open['volume'], pair_cfg_open['pip_size'], m5_pair_key_open)
                self.current_equity += pnl_eod
                self.trade_log.append({
                    "Pair": trade_open['pair_name'], "Timestamp_Open": trade_open['entry_time'], "Type": trade_open['type'],
                    "Entry_Price": trade_open['entry_price_adj'], "SL_Price": trade_open['original_sl_price'], "TP_Price": trade_open.get('tp_price'),
                    "Volume": trade_open['volume'], "Timestamp_Close": last_timestamp, "Exit_Price": last_close_price,
                    "Reason": "End of Backtest", "P&L": pnl_eod, "Equity_After_Trade": self.current_equity
                })
                self.logger.info(f"[{last_timestamp}] {trade_open['pair_name']} {trade_open['type']} EOD CLOSED @ {last_close_price:.5f}. P&L: {pnl_eod:.2f}. Equity: {self.current_equity:.2f}")
            del self.active_trades[m5_pair_key_open]
        
        # Final equity point
        if simulation_timestamps[-1] not in [t[0] for t in self.equity_curve_data]: # Avoid duplicate if already added
            self.equity_curve_data.append((simulation_timestamps[-1], self.current_equity))

        self.data_fetcher.shutdown_mt5() # If MT5 was used by data_fetcher
        self.logger.info("Backtest run finished.")
        return self.generate_report()


    def _calculate_max_drawdown(self, equity_series: pd.Series):
        # (Same as Project 2's _calculate_max_drawdown)
        if equity_series.empty or len(equity_series) < 2: return 0.0, 0.0
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max.replace(0, np.nan)
        drawdown_abs = equity_series - cumulative_max # Absolute drawdown in currency
        
        max_drawdown_percent = drawdown.min() 
        if pd.isna(max_drawdown_percent) or max_drawdown_percent >= 0 : return 0.0, 0.0 # No drawdown or only profit
        
        # Find corresponding absolute drawdown
        max_drawdown_absolute = drawdown_abs[drawdown.idxmin()]
        return abs(max_drawdown_percent) * 100, abs(max_drawdown_absolute)


    def generate_report(self):
        # (Largely same as Project 2's generate_report, ensure paths and column names match)
        self.logger.info("Generating backtest report...")
        file_utils.ensure_dir(self.config.BACKTEST_OUTPUT_DIR)
        
        trade_log_df = pd.DataFrame(self.trade_log)
        if not trade_log_df.empty:
            # Ensure Timestamp_Open and Timestamp_Close are datetime for sorting
            trade_log_df["Timestamp_Open"] = pd.to_datetime(trade_log_df["Timestamp_Open"])
            trade_log_df["Timestamp_Close"] = pd.to_datetime(trade_log_df["Timestamp_Close"])
            trade_log_df.sort_values(by="Timestamp_Close", inplace=True)
            file_utils.save_dataframe_to_csv(trade_log_df, os.path.join(self.config.BACKTEST_OUTPUT_DIR, "trade_log.csv"), index=False)
        else: self.logger.warning("Trade log is empty. No trades to save."); trade_log_df = pd.DataFrame()

        equity_df = pd.DataFrame(self.equity_curve_data, columns=['Timestamp', 'Equity'])
        if not equity_df.empty:
            equity_df.set_index('Timestamp', inplace=True)
            plt.figure(figsize=(14, 7)); plt.plot(equity_df.index, equity_df['Equity'])
            plt.title('Equity Curve'); plt.xlabel('Timestamp'); plt.ylabel(f'Equity ({self.config.HISTORICAL_DATA_SOURCES[self.backtest_pairs_keys[0]].get("backtest_account_currency","USD") if self.backtest_pairs_keys else "USD"})')
            plt.grid(True); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); plt.xticks(rotation=45)
            plt.tight_layout(); plt.savefig(os.path.join(self.config.BACKTEST_OUTPUT_DIR, "equity_curve.png")); plt.close()
        else: self.logger.warning("Equity curve data empty.")

        metrics = {"Initial Capital": f"${self.initial_capital:,.2f}"}
        final_equity = self.current_equity
        metrics["Final Equity"] = f"${final_equity:,.2f}"
        net_pnl = final_equity - self.initial_capital
        metrics["Net Profit/Loss"] = f"${net_pnl:,.2f}"
        metrics["Net Profit/Loss (%)"] = f"{(net_pnl / self.initial_capital * 100):.2f}%" if self.initial_capital != 0 else "N/A"
        
        if not trade_log_df.empty and 'P&L' in trade_log_df.columns and not trade_log_df['P&L'].isna().all():
            closed_trades_df = trade_log_df[trade_log_df['Reason'] != "Open" ].copy() # Exclude "Open" pseudo-logs
            metrics["Total Closed Trades"] = len(closed_trades_df)
            wins = closed_trades_df[closed_trades_df['P&L'] > 0]; losses = closed_trades_df[closed_trades_df['P&L'] < 0]
            metrics.update({
                "Winning Trades": len(wins), "Losing Trades": len(losses),
                "Win Rate (%)": (len(wins) / len(closed_trades_df) * 100) if len(closed_trades_df) > 0 else 0.0,
                "Average Win ($)": wins['P&L'].mean() if not wins.empty else 0.0,
                "Average Loss ($)": losses['P&L'].mean() if not losses.empty else 0.0, # Will be negative
                "Total Gross Profit ($)": wins['P&L'].sum(), 
                "Total Gross Loss ($)": losses['P&L'].sum(), # Sum of negative numbers
                "Profit Factor": (wins['P&L'].sum() / abs(losses['P&L'].sum())) if losses['P&L'].sum() != 0 else np.inf
            })
            if not equity_df.empty:
                max_dd_p, max_dd_a = self._calculate_max_drawdown(equity_df['Equity'])
                metrics.update({"Max Drawdown (%)": f"{max_dd_p:.2f}%", "Max Drawdown ($)": f"${max_dd_a:,.2f}"})
        else: # No trades or P&L column missing/all NaN
            metrics.update({"Total Closed Trades": 0, "Win Rate (%)": 0.0, "Profit Factor": "N/A"})
            if not equity_df.empty and len(equity_df) > 1:
                 max_dd_p, max_dd_a = self._calculate_max_drawdown(equity_df['Equity'])
                 metrics.update({"Max Drawdown (%)": f"{max_dd_p:.2f}%", "Max Drawdown ($)": f"${max_dd_a:,.2f}"})
            else: metrics.update({"Max Drawdown (%)": "0.00%", "Max Drawdown ($)": "$0.00"})

        report_path = os.path.join(self.config.BACKTEST_OUTPUT_DIR, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("Backtest Performance Report (Option A2: Simulated Live Fetch)\n" + "=" * 60 + "\n")
            for k, v_raw in metrics.items():
                # Format numbers nicely
                v_str = str(v_raw)
                if isinstance(v_raw, float) and not ('$' in v_str or '%' in v_str):
                    v_str = f"{v_raw:,.2f}"
                f.write(f"{k}: {v_str}\n")
        self.logger.info(f"Performance report saved to {report_path}")
        self.logger.info("--- Performance Summary ---")
        for k, v in metrics.items(): self.logger.info(f"{k}: {v}")
        self.logger.info("--- End of Summary ---")
        return metrics, trade_log_df, equity_df