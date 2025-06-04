# backtesting/backtester.py
"""
Core backtesting engine for simulating trading strategies on historical data
for multiple currency pairs. Uses pre-computed features and signals.
Implements ATR Stop Loss, dynamic Take Profit, flexible position sizing, and breakeven.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For formatting dates on plots

# Assuming config, logging_utils, file_utils, and RiskManager are accessible
try:
    import config
    from utilities import logging_utils, file_utils
    # RiskManager is passed in __init__
except ImportError:
    print("FATAL: Could not perform standard imports in Backtester. Ensure paths are correct.")
    raise

# logger will be passed in __init__

class Backtester:
    def __init__(self, config_obj, logger_obj, initial_capital: float,
                 all_featured_data: dict, # {m5_pair_key: pd.DataFrame_with_all_features}
                 all_final_signals: dict, # {m5_pair_key: pd.Series_of_signals}
                 risk_manager_instance):

        self.config = config_obj
        self.logger = logger_obj

        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.spread_pips = self.config.BACKTEST_SPREAD_PIPS

        self.all_featured_data = all_featured_data
        self.all_final_signals = all_final_signals
        self.risk_manager = risk_manager_instance

        # Strategy parameters from config
        self.use_breakeven = getattr(self.config, 'USE_BREAKEVEN', True)
        self.breakeven_trigger_pips = getattr(self.config, 'BREAKEVEN_TRIGGER_PIPS', 20)
        self.breakeven_adjust_pips = getattr(self.config, 'BREAKEVEN_ADJUST_PIPS', 2)
        self.min_tp_pips_execution = getattr(self.config, 'MIN_TP_PIPS_EXECUTION', 10) # If defined, else fallback

        self.backtest_pairs_keys = list(self.all_featured_data.keys()) # Pairs to backtest

        self.pair_configs_details = {} # Store pip_size, etc. for quick lookup
        for key in self.backtest_pairs_keys:
            cfg_data = self.config.HISTORICAL_DATA_SOURCES.get(key)
            if cfg_data:
                self.pair_configs_details[key] = {
                    "pair_name": cfg_data['pair'],
                    "pip_size": cfg_data['pip_size'],
                    # Backtest sizing params (already in RiskManager via config, but good to have pip_size here)
                }
            else:
                self.logger.error(f"Backtester: Config for M5 pair key '{key}' not found. It might be skipped if not in all_featured_data.")
        # Filter again in case all_featured_data didn't have all keys from PRIMARY_MODEL_PAIRS_TIMEFRAMES
        self.backtest_pairs_keys = [k for k in self.backtest_pairs_keys if k in self.pair_configs_details]


        self.trade_log = []
        self.equity_curve_data = []
        self.active_trades = {} # {m5_pair_key: trade_dict}

        self.logger.info(f"Backtester initialized (Pre-computed Features Mode). Capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Backtesting M5 primary pairs: {self.backtest_pairs_keys}")

    def _get_pair_config(self, m5_pair_key):
        return self.pair_configs_details.get(m5_pair_key)

    def _apply_spread_to_entry(self, price, trade_type, pip_size):
        spread_amount = self.spread_pips * pip_size
        return price + spread_amount if trade_type == 'BUY' else price - spread_amount

    def _calculate_pnl(self, entry_price_adj, exit_price, trade_type, lot_size, pip_size, m5_pair_key):
        pair_cfg = self._get_pair_config(m5_pair_key)
        # P&L calculation now relies on RiskManager's helper or direct calculation using point value
        # For simplicity, using pip_size and assuming a point_value_per_pip_per_lot
        # This needs to be robust using the same logic as RiskManager for consistency if possible
        # Or, store point_value_per_pip_per_lot in pair_configs_details
        
        # Simplified P&L for now, assuming point_value_per_lot is for 1 pip
        # A more accurate way is to use the point size and trade_tick_value from config
        # as used in RiskManager._calculate_position_size_dynamic
        
        price_diff_pips = ((exit_price - entry_price_adj) / pip_size) if trade_type == 'BUY' else ((entry_price_adj - exit_price) / pip_size)
        
        # Mocking point_value_per_pip_per_lot for this example, should come from config
        point_value_per_pip_per_lot = 10.0 # Example for EURUSD on a standard account (1 lot = $10 per pip)
        if pair_cfg and 'backtest_mt5_trade_tick_value' in pair_cfg and 'backtest_mt5_point' in pair_cfg and pair_cfg['backtest_mt5_point'] > 0:
            # Value of 1 pip for 1 lot = (pip_size / point_size) * tick_value_for_1_lot
            point_value_per_pip_per_lot = (pip_size / pair_cfg['backtest_mt5_point']) * pair_cfg['backtest_mt5_trade_tick_value']

        pnl = price_diff_pips * point_value_per_pip_per_lot * lot_size
        return pnl

    def _determine_dynamic_tp(self, entry_price_adj: float, trade_type: str, current_bar_features: pd.Series, pip_size: float) -> float | None:
        """Determines TP based on H1 dynamic SNR, then M5 pivots."""
        tp_price = None
        tp_source = "None"
        min_tp_dist_points = getattr(self.config, 'MIN_TP_PIPS_EXECUTION', 10) * pip_size

        # Check H1 Dynamic SNR levels
        h1_levels_to_check_s = self.config.H1_DYNAMIC_SUPPORT_COLS
        h1_levels_to_check_r = self.config.H1_DYNAMIC_RESISTANCE_COLS
        
        potential_tps = []
        if trade_type == 'BUY':
            for col in h1_levels_to_check_r:
                level = current_bar_features.get(col)
                if pd.notna(level) and level > entry_price_adj + min_tp_dist_points:
                    potential_tps.append(level)
            if potential_tps: tp_price = min(potential_tps); tp_source = "H1 Dyn Resistance"
        else: # SELL
            for col in h1_levels_to_check_s:
                level = current_bar_features.get(col)
                if pd.notna(level) and level < entry_price_adj - min_tp_dist_points:
                    potential_tps.append(level)
            if potential_tps: tp_price = max(potential_tps); tp_source = "H1 Dyn Support"

        # Fallback to M5 Pivots if no H1 TP
        if tp_price is None:
            m5_sup_col = self.config.M5_PIVOT_SUPPORT_COL
            m5_res_col = self.config.M5_PIVOT_RESISTANCE_COL
            if trade_type == 'BUY' and pd.notna(current_bar_features.get(m5_res_col)):
                level = current_bar_features.get(m5_res_col)
                if level > entry_price_adj + min_tp_dist_points:
                    tp_price = level; tp_source = "M5 Pivot Resistance"
            elif trade_type == 'SELL' and pd.notna(current_bar_features.get(m5_sup_col)):
                level = current_bar_features.get(m5_sup_col)
                if level < entry_price_adj - min_tp_dist_points:
                    tp_price = level; tp_source = "M5 Pivot Support"
        
        if tp_price: self.logger.debug(f"Dynamic TP for {trade_type} set to {tp_price:.5f} (Source: {tp_source})")
        return tp_price


    def run_backtest(self):
        import logging # Import logging here
        self.logger.info("Starting backtest run (Pre-computed Features & Signals Mode)...")
        
        # Create a unified timeline from all M5 data indices
        # Use union to combine DatetimeIndex objects
        unified_timestamps = pd.DatetimeIndex([])
        for m5_pair_key in self.backtest_pairs_keys:
            if m5_pair_key in self.all_featured_data:
                unified_timestamps = unified_timestamps.union(self.all_featured_data[m5_pair_key].index)

        if unified_timestamps.empty:
            self.logger.error("No featured data available for any pair or unified timestamp index is empty. Aborting backtest."); return None,None,None

        unified_timestamps = unified_timestamps.sort_values()
        self.equity_curve_data.append((unified_timestamps[0] - pd.Timedelta(minutes=5), self.initial_capital)) # Equity before first bar

        for current_timestamp in unified_timestamps:
            self.logger.log(logging.DEBUG - 5, f"--- Processing Timestamp: {current_timestamp} ---") # Very verbose
            self.equity_curve_data.append((current_timestamp, self.current_equity))

            for m5_pair_key in self.backtest_pairs_keys:
                if m5_pair_key not in self.all_featured_data or \
                   m5_pair_key not in self.all_final_signals or \
                   current_timestamp not in self.all_featured_data[m5_pair_key].index or \
                   current_timestamp not in self.all_final_signals[m5_pair_key].index:
                    continue # Data or signal not available for this pair at this timestamp

                pair_cfg = self._get_pair_config(m5_pair_key)
                if not pair_cfg: continue 
                pair_name = pair_cfg['pair_name']
                pip_size = pair_cfg['pip_size']

                current_bar_features = self.all_featured_data[m5_pair_key].loc[current_timestamp]
                # OHLC for current bar (used for SL/TP checks)
                bar_open = current_bar_features[self.config.OPEN_COL]
                bar_high = current_bar_features[self.config.HIGH_COL]
                bar_low = current_bar_features[self.config.LOW_COL]
                bar_close = current_bar_features[self.config.CLOSE_COL]

                # Manage open trade for this pair
                trade_closed_this_bar = False
                if m5_pair_key in self.active_trades:
                    trade = self.active_trades[m5_pair_key]
                    exit_price, reason = None, None
                    
                    # Breakeven
                    if self.use_breakeven and not trade.get('is_breakeven', False):
                        profit_pips_be = 0
                        if trade['type'] == 'BUY': profit_pips_be = (bar_high - trade['entry_price_adj']) / pip_size
                        else: profit_pips_be = (trade['entry_price_adj'] - bar_low) / pip_size
                        
                        if profit_pips_be >= self.breakeven_trigger_pips:
                            new_sl_be = trade['entry_price_adj'] + (self.breakeven_adjust_pips * pip_size if trade['type'] == 'BUY' else -self.breakeven_adjust_pips * pip_size)
                            if (trade['type'] == 'BUY' and new_sl_be > trade['sl_price']) or \
                               (trade['type'] == 'SELL' and new_sl_be < trade['sl_price']):
                                trade['sl_price'] = new_sl_be; trade['is_breakeven'] = True
                                self.logger.info(f"[{current_timestamp}] {pair_name} {trade['type']} Breakeven. New SL: {new_sl_be:.5f}")
                    
                    # SL/TP Check
                    if trade['type'] == 'BUY':
                        if bar_low <= trade['sl_price']: exit_price, reason = trade['sl_price'], "Stop Loss"
                        elif trade.get('tp_price') and bar_high >= trade['tp_price']: exit_price, reason = trade['tp_price'], "Take Profit"
                    else: # SELL
                        if bar_high >= trade['sl_price']: exit_price, reason = trade['sl_price'], "Stop Loss"
                        elif trade.get('tp_price') and bar_low <= trade['tp_price']: exit_price, reason = trade['tp_price'], "Take Profit"
                    
                    # Fallback EMA Exit (if no TP was set initially and SL/TP not hit)
                    if exit_price is None and not trade.get('tp_price'):
                        m5_exit_ema_val = current_bar_features.get(self.config.M5_EXIT_EMA_COL)
                        if pd.notna(m5_exit_ema_val):
                            if (trade['type'] == 'BUY' and bar_close < m5_exit_ema_val) or \
                               (trade['type'] == 'SELL' and bar_close > m5_exit_ema_val):
                                exit_price, reason = bar_close, f"M5 EMA Exit ({self.config.M5_EXIT_EMA_PERIOD})"

                    if exit_price is not None:
                        pnl = self._calculate_pnl(trade['entry_price_adj'], exit_price, trade['type'], trade['volume'], pip_size, m5_pair_key)
                        self.current_equity += pnl
                        self.trade_log.append({
                            "Pair": pair_name, "Timestamp_Open": trade['entry_time'], "Type": trade['type'],
                            "Entry_Price": trade['entry_price_adj'], "SL_Price": trade['original_sl_price'], "TP_Price": trade.get('tp_price'),
                            "Volume": trade['volume'], "Timestamp_Close": current_timestamp, "Exit_Price": exit_price, 
                            "Reason": reason, "P&L": pnl, "Equity_After_Trade": self.current_equity
                        })
                        self.logger.info(f"[{current_timestamp}] {pair_name} {trade['type']} CLOSED by {reason} @ {exit_price:.5f}. P&L: {pnl:.2f}. Equity: {self.current_equity:.2f}")
                        del self.active_trades[m5_pair_key]; trade_closed_this_bar = True

                # New Trade Entry
                if not trade_closed_this_bar and m5_pair_key not in self.active_trades:
                    current_signal = self.all_final_signals[m5_pair_key].loc[current_timestamp]
                    if current_signal != 0: # BUY (1) or SELL (-1)
                        signal_type = 'BUY' if current_signal == 1 else 'SELL'
                        
                        # Entry at open of current bar (signal was from previous bar's close features)
                        entry_price_raw = bar_open
                        entry_price_adj = self._apply_spread_to_entry(entry_price_raw, signal_type, pip_size)
                        
                        atr_val_for_sl = current_bar_features.get(self.config.M5_ATR_COL_BASE)
                        if pd.isna(atr_val_for_sl) or atr_val_for_sl <= 0:
                            self.logger.warning(f"[{current_timestamp}] Invalid ATR ({atr_val_for_sl}) for SL for {pair_name}. Skipping trade."); continue

                        sl_price = self.risk_manager.calculate_stop_loss(entry_price_adj, atr_val_for_sl, signal_type, pair_name)
                        
                        # Determine TP: First try dynamic H1/M5 pivots, then fallback to RRR
                        tp_price_dynamic = self._determine_dynamic_tp(entry_price_adj, signal_type, current_bar_features, pip_size)
                        tp_price_rrr = self.risk_manager.calculate_take_profit(entry_price_adj, sl_price, signal_type, pair_name)
                        
                        tp_price = tp_price_dynamic if tp_price_dynamic is not None else tp_price_rrr # Prioritize dynamic

                        if sl_price is None: self.logger.warning(f"SL calc failed for {pair_name}. Skipping."); continue
                        # Basic validation
                        if (signal_type == 'BUY' and (sl_price >= entry_price_adj or (tp_price and tp_price <= entry_price_adj))) or \
                           (signal_type == 'SELL' and (sl_price <= entry_price_adj or (tp_price and tp_price >= entry_price_adj))):
                            self.logger.warning(f"[{current_timestamp}] Invalid SL/TP for {pair_name} {signal_type}: E={entry_price_adj:.5f} SL={sl_price:.5f} TP={tp_price}. Skipping.")
                            continue

                        trade_volume = self.risk_manager.get_trade_volume(
                            account_balance=self.current_equity, entry_price=entry_price_adj, 
                            sl_price=sl_price, symbol=pair_name, symbol_info_mt5=None
                        )
                        if trade_volume <= 0: self.logger.error(f"Volume is {trade_volume} for {pair_name}. Skipping."); continue

                        self.active_trades[m5_pair_key] = {
                            'type': signal_type, 'entry_time': current_timestamp,
                            'entry_price_raw': entry_price_raw, 'entry_price_adj': entry_price_adj,
                            'sl_price': sl_price, 'original_sl_price': sl_price, 'tp_price': tp_price,
                            'volume': trade_volume, 'is_breakeven': False, 'pair_name': pair_name
                        }
                        self.logger.info(f"[{current_timestamp}] {pair_name} OPENED {signal_type} @ {entry_price_adj:.5f} (Raw:{entry_price_raw:.5f}), Vol:{trade_volume:.2f}, SL:{sl_price:.5f}, TP:{tp_price if tp_price else 'EMA Exit'}")
            # End pair loop
        # End timestamp loop

        self.logger.info("End of backtest simulation period. Closing remaining open trades...")
        # (Closing logic for EOD trades - same as before)
        for m5_pair_key_open, trade_open in list(self.active_trades.items()):
            pair_cfg_open = self._get_pair_config(m5_pair_key_open)
            if m5_pair_key_open not in self.all_featured_data or self.all_featured_data[m5_pair_key_open].empty: continue
            
            last_bar_features = self.all_featured_data[m5_pair_key_open].iloc[-1]
            last_close_price = last_bar_features[self.config.CLOSE_COL]
            last_timestamp = self.all_featured_data[m5_pair_key_open].index[-1]

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

        if not unified_timestamps.empty and (not self.equity_curve_data or self.equity_curve_data[-1][0] != unified_timestamps[-1]):
             self.equity_curve_data.append((unified_timestamps[-1], self.current_equity))
        elif not unified_timestamps.empty and self.equity_curve_data and self.equity_curve_data[-1][0] == unified_timestamps[-1]:
             self.equity_curve_data[-1] = (unified_timestamps[-1], self.current_equity) # Update last point

        self.logger.info("Backtest run finished.")
        return self.generate_report()

    def generate_report(self):
        # (This method remains largely the same as the one generated previously for Project 2's backtester)
        # Ensure it uses self.config for paths and handles empty trade_log_df or equity_df.
        self.logger.info("Generating backtest report...")
        file_utils.ensure_dir(self.config.BACKTEST_OUTPUT_DIR)
        
        trade_log_df = pd.DataFrame(self.trade_log)
        if not trade_log_df.empty:
            trade_log_df["Timestamp_Open"] = pd.to_datetime(trade_log_df["Timestamp_Open"])
            trade_log_df["Timestamp_Close"] = pd.to_datetime(trade_log_df["Timestamp_Close"])
            trade_log_df.sort_values(by="Timestamp_Close", inplace=True)
            file_utils.save_dataframe_to_csv(trade_log_df, os.path.join(self.config.BACKTEST_OUTPUT_DIR, "trade_log.csv"), index=False)
        else: self.logger.warning("Trade log is empty."); trade_log_df = pd.DataFrame() 

        equity_df = pd.DataFrame(self.equity_curve_data, columns=['Timestamp', 'Equity'])
        if not equity_df.empty and len(equity_df) > 1: # Need at least 2 points to plot
            equity_df.set_index('Timestamp', inplace=True)
            plt.figure(figsize=(14, 7)); plt.plot(equity_df.index, equity_df['Equity'])
            plt.title('Equity Curve'); plt.xlabel('Timestamp'); 
            currency_label = "USD" # Default
            if self.backtest_pairs_keys and self.config.HISTORICAL_DATA_SOURCES.get(self.backtest_pairs_keys[0]):
                currency_label = self.config.HISTORICAL_DATA_SOURCES[self.backtest_pairs_keys[0]].get("backtest_account_currency", "USD")
            plt.ylabel(f'Equity ({currency_label})'); 
            plt.grid(True); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); plt.xticks(rotation=45)
            plt.tight_layout(); plt.savefig(os.path.join(self.config.BACKTEST_OUTPUT_DIR, "equity_curve.png")); plt.close()
        elif not equity_df.empty: # Only one point (initial capital)
             self.logger.warning("Equity curve has only one data point (initial capital). Plot not generated.")
        else: self.logger.warning("Equity curve data empty. Plot not generated.")

        metrics = {"Initial Capital": f"${self.initial_capital:,.2f}"}
        final_equity = self.current_equity 
        metrics["Final Equity"] = f"${final_equity:,.2f}"
        net_pnl = final_equity - self.initial_capital
        metrics["Net Profit/Loss"] = f"${net_pnl:,.2f}"
        metrics["Net Profit/Loss (%)"] = f"{(net_pnl / self.initial_capital * 100):.2f}%" if self.initial_capital != 0 else "N/A"
        
        # Filter out "Open" pseudo-logs if they were added, for metric calculation
        closed_trades_df = trade_log_df[trade_log_df['Reason'] != "Open" ].copy() if "Reason" in trade_log_df else trade_log_df.copy()
        
        if not closed_trades_df.empty and 'P&L' in closed_trades_df.columns and not closed_trades_df['P&L'].isna().all():
            metrics["Total Closed Trades"] = len(closed_trades_df)
            wins = closed_trades_df[closed_trades_df['P&L'] > 0]; losses = closed_trades_df[closed_trades_df['P&L'] < 0]
            metrics.update({
                "Winning Trades": len(wins), "Losing Trades": len(losses),
                "Win Rate (%)": (len(wins) / len(closed_trades_df) * 100) if len(closed_trades_df) > 0 else 0.0,
                "Average Win ($)": wins['P&L'].mean() if not wins.empty else 0.0,
                "Average Loss ($)": losses['P&L'].mean() if not losses.empty else 0.0,
                "Total Gross Profit ($)": wins['P&L'].sum(), 
                "Total Gross Loss ($)": losses['P&L'].sum(), # Sum of negative numbers
                "Profit Factor": (wins['P&L'].sum() / abs(losses['P&L'].sum())) if losses['P&L'].sum() != 0 and wins['P&L'].sum() > 0 else (np.inf if wins['P&L'].sum() > 0 else 0)
            })
            if not equity_df.empty and len(equity_df) > 1:
                max_dd_p, max_dd_a = self._calculate_max_drawdown(equity_df['Equity'])
                metrics.update({"Max Drawdown (%)": f"{max_dd_p:.2f}%", "Max Drawdown ($)": f"${max_dd_a:,.2f}"})
            else: metrics.update({"Max Drawdown (%)": "0.00%", "Max Drawdown ($)": "$0.00"})
        else:
            metrics.update({"Total Closed Trades": 0, "Win Rate (%)": 0.0, "Profit Factor": "N/A"})
            metrics.update({"Max Drawdown (%)": "0.00%", "Max Drawdown ($)": "$0.00"})

        report_path = os.path.join(self.config.BACKTEST_OUTPUT_DIR, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("Backtest Performance Report (Pre-computed Features Mode)\n" + "=" * 60 + "\n")
            for k, v_raw in metrics.items():
                v_str = str(v_raw)
                if isinstance(v_raw, float) and not ('$' in v_str or '%' in v_str or 'inf' in v_str.lower()):
                    v_str = f"{v_raw:,.2f}"
                f.write(f"{k}: {v_str}\n")
        self.logger.info(f"Performance report saved to {report_path}")
        self.logger.info("--- Performance Summary ---")
        for k, v in metrics.items(): self.logger.info(f"{k}: {v}")
        self.logger.info("--- End of Summary ---")
        return metrics, trade_log_df, equity_df