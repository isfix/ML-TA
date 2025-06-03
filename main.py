# main.py
"""
Main entry point for the AI-ML Trading Bot.
Handles different modes of operation: train, backtest, live.
Implements a multi-timeframe strategy with dynamic H1 SNR, M5 ML features & entry.
ML models are tuned with Optuna and can be ensembled.
Risk management is flexible (fixed lot or dynamic percentage).
Backtesting simulates live data fetching from MT5.
"""
import argparse
import os
import pandas as pd
import numpy as np
import time
import shutil
import logging # For initial logger setup before config is fully loaded by setup_environment
from datetime import datetime, timezone, timedelta

# Attempt to import config first, it sets up paths that utilities might use
try:
    import config
except ImportError as e:
    print(f"FATAL ERROR: Could not import 'config.py'. Ensure it exists and is in the Python path. Error: {e}")
    exit(1)
except Exception as e: # Catch any other error during config import
    print(f"FATAL ERROR: Unexpected error importing 'config.py': {e}")
    exit(1)

# Now import other modules
from utilities import logging_utils, file_utils # Project 2 style
from data_ingestion.data_fetcher import DataFetcher # Project 1 style
from feature_engineering.market_structure import MarketStructureAnalyzer # Project 2 style, adapted for dynamic SNR
from feature_engineering.feature_builder import FeatureBuilder # Project 1 style, adapted into a class
from ml_models.model_manager import ModelManager # Project 1 style, with Optuna/Ensemble
from signal_generation.signal_processor import SignalProcessor # Project 1 style
from execution.mt5_interface import MT5Interface # Project 1 style
from execution.risk_manager import RiskManager # Project 1 style, with flexible sizing
from backtesting.backtester import Backtester # Project 2 style, adapted for MT5 data & P1 logic

# Global logger, will be configured by setup_environment
# For now, a basic configuration for messages before setup_environment completes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger("MainApp")


def setup_environment():
    """Loads configuration (already done by import), sets up logging, and creates necessary directories."""
    # Configuration is already loaded by `import config`.
    # Setup logging using the utility function from Project 1, adapted for Project 2's config variables
    # The logger instance returned and configured here will be the root logger if not already set.
    # We'll reconfigure the 'MainApp' logger specifically.
    global app_logger # Allow modification of the global app_logger
    app_logger = logging_utils.setup_logger(
        'MainApp',
        config.LOG_FILE_APP,
        level_str=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'INFO'
    )
    app_logger.info("Environment setup: Logging initialized.")

    # Create essential directories (Project 1 style, paths from config)
    # config.py already creates these, but this is a good place to confirm or log.
    for dir_path_attr_name, dir_path_val in [
        ('DATA_DIR', config.DATA_DIR),
        ('MODELS_DIR', config.MODELS_DIR),
        ('LOG_DIR', config.LOG_DIR),
        ('REPORTS_DIR', config.REPORTS_DIR),
        ('BACKTEST_OUTPUT_DIR', config.BACKTEST_OUTPUT_DIR)
    ]:
        if not os.path.exists(dir_path_val):
            try:
                os.makedirs(dir_path_val)
                app_logger.info(f"Created directory: {dir_path_val}")
            except OSError as e:
                app_logger.error(f"Failed to create directory {dir_path_val}: {e}", exc_info=True)
        else:
            app_logger.debug(f"Directory already exists: {dir_path_val}")
    app_logger.info("Environment setup: Directories checked/created.")
    return True # Indicate success

def clear_directory_contents(directory_path, dir_name_for_log):
    """Safely clears contents of a specified directory (from Project 2)."""
    if not os.path.exists(directory_path):
        app_logger.info(f"{dir_name_for_log} directory ({directory_path}) does not exist. Nothing to clear.")
        return
    if not os.path.isdir(directory_path):
        app_logger.error(f"Path {directory_path} is not a directory. Cannot clear.")
        return

    app_logger.info(f"Attempting to clear contents of {dir_name_for_log} directory: {directory_path}")
    cleared_count = 0; error_count = 0
    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path): os.unlink(item_path)
            elif os.path.isdir(item_path): shutil.rmtree(item_path)
            cleared_count += 1
        except Exception as e:
            app_logger.error(f"Failed to delete {item_path}. Reason: {e}"); error_count += 1
    if cleared_count > 0: app_logger.info(f"Successfully cleared {cleared_count} items from {directory_path}.")
    if error_count > 0: app_logger.warning(f"Failed to clear {error_count} items from {directory_path}.")
    if cleared_count == 0 and error_count == 0: app_logger.info(f"{dir_name_for_log} directory ({directory_path}) was empty or no clearable items.")


def run_training(use_smote_arg=False):
    app_logger.info(f"--- Starting Training Mode --- SMOTE CLI: {use_smote_arg}")
    training_logger = logging_utils.setup_logger('TrainingProcess', config.LOG_FILE_TRAINING, level_str=config.LOG_LEVEL)
    data_fetcher = DataFetcher(config, training_logger) # P1 DataFetcher

    if not config.PRIMARY_MODEL_PAIRS_TIMEFRAMES:
        training_logger.error("No M5 pairs in config.PRIMARY_MODEL_PAIRS_TIMEFRAMES. Aborting training.")
        return

    for m5_pair_key in config.PRIMARY_MODEL_PAIRS_TIMEFRAMES:
        training_logger.info(f"===== Training for M5 Pair: {m5_pair_key} =====")
        m5_pair_config = config.HISTORICAL_DATA_SOURCES.get(m5_pair_key)
        if not m5_pair_config or m5_pair_config.get('timeframe_str', '').upper() != 'M5':
            training_logger.error(f"Config for M5 pair key '{m5_pair_key}' is invalid or not found. Skipping."); continue

        pair_name = m5_pair_config['pair']
        h1_pair_key = f"{pair_name}_{config.TIMEFRAME_H1_STR}" # e.g., EURUSD_H1
        h1_pair_config = config.HISTORICAL_DATA_SOURCES.get(h1_pair_key)

        # Fetch data (P1 DataFetcher logic)
        try:
            start_dt = datetime.fromisoformat(config.START_DATE_HISTORICAL).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(config.END_DATE_HISTORICAL).replace(tzinfo=timezone.utc)
        except ValueError as e:
            training_logger.error(f"Invalid date format in config: {e}. Use YYYY-MM-DD. Skipping {m5_pair_key}."); continue

        training_logger.info(f"Fetching M5 data for {pair_name} ({m5_pair_config['mt5_timeframe']})...")
        df_m5_raw = data_fetcher.fetch_historical_data(pair_name, m5_pair_config['mt5_timeframe'], start_dt, end_dt)
        if df_m5_raw is None or df_m5_raw.empty:
            training_logger.error(f"No M5 data for {pair_name}. Skipping training for this pair."); continue

        df_h1_raw = None
        if h1_pair_config:
            training_logger.info(f"Fetching H1 data for {pair_name} ({h1_pair_config['mt5_timeframe']})...")
            df_h1_raw = data_fetcher.fetch_historical_data(pair_name, h1_pair_config['mt5_timeframe'], start_dt, end_dt)
            if df_h1_raw is None or df_h1_raw.empty:
                training_logger.warning(f"No H1 data for {pair_name}. H1 context features will be limited.")
        else:
            training_logger.warning(f"No H1 config for {pair_name}. H1 context features will be limited.")
        
        data_fetcher.shutdown_mt5() # Shutdown after fetching for this pair if MT5 source

        # Feature Engineering (P1 logic via FeatureBuilder class)
        try:
            # FeatureBuilder needs to be adapted to take m5_df, h1_df, config, logger
            # And its methods called sequentially.
            # For now, assuming FeatureBuilder class encapsulates P1's feature_builder.py logic
            market_analyzer = MarketStructureAnalyzer(config, training_logger) # For dynamic SNR
            feature_builder = FeatureBuilder(config, training_logger, market_analyzer)

            training_logger.info("Building features and labels...")
            # build_features_and_labels will call internal methods for indicators, H1 context, M5 features, and labeling
            features_df_labeled = feature_builder.build_features_and_labels(df_m5_raw, df_h1_raw)

        except Exception as e:
            training_logger.error(f"Error during feature engineering for {pair_name}: {e}", exc_info=True); continue

        if features_df_labeled.empty or config.TARGET_COLUMN_NAME not in features_df_labeled.columns:
            training_logger.error(f"Feature or target column '{config.TARGET_COLUMN_NAME}' not generated for {pair_name}. Skipping."); continue
        
        # ML Model Training (P1 ModelManager with Optuna/Ensemble)
        # Suffix for model files should include the pair key
        model_manager = ModelManager(config, training_logger, model_name_prefix=f"{pair_name}_{m5_pair_config['timeframe_str']}")
        
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = model_manager.prepare_data_for_model(features_df_labeled)
        if X_train is None or X_train.empty:
            training_logger.error(f"Data preparation failed for {pair_name}. Skipping model training."); continue

        trained_models_info = [] # List of dicts: {'name': model_type, 'model': instance, 'best_params': params}
        for model_type_name in config.ML_MODEL_TYPES:
            training_logger.info(f"--- Optimizing and Training {model_type_name} for {pair_name} ---")
            best_params, best_score = None, None
            if config.USE_OPTUNA:
                training_logger.info(f"Running Optuna for {model_type_name}...")
                best_params, best_score = model_manager.optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type_name)
                training_logger.info(f"Optuna best params for {model_type_name}: {best_params}, Best score: {best_score:.4f}")

            training_logger.info(f"Training final {model_type_name} with {'best' if best_params else 'default'} params...")
            # Train on full training data (or train+val) with best_params
            # For simplicity, let's assume train_individual_model handles using best_params if available
            final_model_instance = model_manager.train_individual_model(
                X_train, y_train, X_val, y_val, # Pass val for evaluation within
                model_type_name,
                model_params_override=best_params # Pass Optuna's best params
            )

            if final_model_instance:
                model_filename_prefix_individual = f"{pair_name}_{m5_pair_config['timeframe_str']}_{model_type_name}{'_tuned' if best_params else ''}"
                model_manager.save_model_and_scaler(final_model_instance, scaler, model_filename_prefix_individual, model_manager.feature_names_)
                trained_models_info.append({'name': model_type_name, 'model': final_model_instance, 'params': best_params or 'default'})
        
        if not trained_models_info:
            training_logger.error(f"No base models trained successfully for {pair_name}."); continue

        if config.TRAIN_ENSEMBLE_MODEL and len(trained_models_info) > 1:
            training_logger.info(f"--- Training Ensemble Model for {pair_name} ---")
            # Pass only the model instances to train_ensemble
            base_model_instances_for_ensemble = [{'name': info['name'], 'model': info['model']} for info in trained_models_info]
            ensemble_model_instance = model_manager.train_ensemble(X_train, y_train, X_val, y_val, base_model_instances_for_ensemble)
            if ensemble_model_instance:
                ensemble_filename_prefix = f"{pair_name}_{m5_pair_config['timeframe_str']}_ensemble"
                model_manager.save_model_and_scaler(ensemble_model_instance, scaler, ensemble_filename_prefix, model_manager.feature_names_)
                training_logger.info(f"Ensemble model for {pair_name} trained and saved.")
            else:
                training_logger.warning(f"Failed to train ensemble model for {pair_name}.")
        
        training_logger.info(f"===== Finished training for M5 Pair: {m5_pair_key} =====")
    app_logger.info("--- Training Mode Finished ---")


def run_backtesting():
    app_logger.info("--- Starting Backtesting Mode (Option A2: Incremental MT5 Data Simulation) ---")
    backtest_logger = logging_utils.setup_logger('BacktestProcess', config.LOG_FILE_BACKTEST, level_str=config.LOG_LEVEL)
    
    data_fetcher = DataFetcher(config, backtest_logger)
    risk_manager = RiskManager(config, backtest_logger)
    market_analyzer = MarketStructureAnalyzer(config, backtest_logger) # For dynamic SNR
    
    # Function to load model for a specific pair (closure)
    def model_loader(pair_key_for_model):
        # Determine model file prefix (ensemble or best tuned individual)
        # Example: using MODEL_TYPE_FOR_TRADING from config
        pair_cfg = config.HISTORICAL_DATA_SOURCES.get(pair_key_for_model)
        if not pair_cfg: return None, None, None
        
        # Construct prefix based on MODEL_TYPE_FOR_TRADING
        # e.g., if MODEL_TYPE_FOR_TRADING = 'XGBClassifier_ensemble'
        # prefix becomes 'EURUSD_M5_XGBClassifier_ensemble'
        model_file_prefix = f"{pair_cfg['pair']}_{pair_cfg['timeframe_str']}_{config.MODEL_TYPE_FOR_TRADING}"
        
        # If MODEL_TYPE_FOR_TRADING implies a specific single model (e.g., 'XGBClassifier_tuned')
        # model_file_prefix = f"{pair_cfg['pair']}_{pair_cfg['timeframe_str']}_{config.MODEL_TYPE_FOR_TRADING}"

        mm = ModelManager(config, backtest_logger, model_name_prefix=model_file_prefix) # Prefix used for loading
        return mm.load_model_and_scaler(model_file_prefix) # load_model_and_scaler uses its internal prefix logic

    # Function to create SignalProcessor instance
    def signal_processor_creator(loaded_model, loaded_scaler, loaded_feature_names):
        return SignalProcessor(config, backtest_logger, loaded_model, loaded_scaler, loaded_feature_names)

    backtester = Backtester(
        config=config,
        logger=backtest_logger,
        data_fetcher=data_fetcher,
        market_analyzer=market_analyzer, # Pass market_analyzer
        model_loader_func=model_loader,
        signal_processor_creator_func=signal_processor_creator,
        risk_manager=risk_manager
    )

    backtest_logger.info("Running backtest simulation...")
    try:
        performance_metrics, trade_log_df, equity_curve_df = backtester.run_backtest()
        if performance_metrics:
            backtest_logger.info("Backtest finished. Performance metrics generated.")
            # Reporting is handled by backtester.generate_report()
        else:
            backtest_logger.warning("Backtest completed, but no performance metrics were generated (likely no trades).")
    except Exception as e:
        backtest_logger.error(f"Critical error during backtest execution: {e}", exc_info=True)
    finally:
        data_fetcher.shutdown_mt5() # Ensure MT5 connection is closed if used

    app_logger.info("--- Backtesting Mode Finished ---")


def run_live_trading():
    app_logger.info("--- Starting Live Trading Mode ---")
    live_logger = logging_utils.setup_logger('LiveTradingProcess', config.LOG_FILE_LIVE, level_str=config.LOG_LEVEL)
    
    mt5_interface = MT5Interface(config, live_logger)
    if not mt5_interface.is_connected: # is_connected flag from P1 MT5Interface
        live_logger.error("Failed to connect to MT5. Live trading cannot start."); return

    data_fetcher = DataFetcher(config, live_logger) # Will use MT5
    risk_manager = RiskManager(config, live_logger)
    market_analyzer = MarketStructureAnalyzer(config, live_logger)
    
    # Store loaded models and signal processors per pair to avoid reloading every cycle
    live_models_scalers_features = {} # {pair_key: (model, scaler, features_names)}
    live_signal_processors = {}      # {pair_key: SignalProcessor_instance}

    try:
        while True:
            now_utc = datetime.now(timezone.utc)
            # Calculate time until the next M5 candle (logic from P1 main.py)
            next_m5_boundary_minute = (now_utc.minute // 5 + 1) * 5
            if next_m5_boundary_minute >= 60:
                next_m5_target_time = now_utc.replace(hour=(now_utc.hour + 1) % 24, minute=0, second=config.LIVE_TRADING_POLL_INTERVAL_SECONDS, microsecond=0)
                if now_utc.hour == 23: next_m5_target_time += timedelta(days=1)
            else:
                next_m5_target_time = now_utc.replace(minute=next_m5_boundary_minute, second=config.LIVE_TRADING_POLL_INTERVAL_SECONDS, microsecond=0)
            
            sleep_duration = (next_m5_target_time - now_utc).total_seconds()
            if sleep_duration <= 0: sleep_duration += 300 # Add 5 mins if already past

            live_logger.info(f"Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}. Waiting {sleep_duration:.0f}s for next M5 cycle.")
            time.sleep(max(1.0, sleep_duration))
            
            live_logger.info(f"--- Live Trading Cycle Start: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} ---")

            # Manage existing trades first (e.g., trailing stops, breakeven)
            open_positions = mt5_interface.get_open_positions(magic_number=config.MT5_MAGIC_NUMBER)
            if open_positions:
                live_logger.info(f"Managing {len(open_positions)} open trade(s)...")
                for pos in open_positions:
                    # TODO: Implement Trailing Stop / Breakeven logic from RiskManager/Config
                    # Example:
                    # if config.USE_BREAKEVEN:
                    #   current_price = mt5_interface.get_symbol_tick_info(pos.symbol).ask if pos.type == mt5.ORDER_TYPE_SELL else mt5_interface.get_symbol_tick_info(pos.symbol).bid
                    #   risk_manager.check_and_apply_breakeven(pos, current_price, mt5_interface)
                    # if config.USE_TRAILING_STOP:
                    #   risk_manager.check_and_apply_trailing_stop(pos, current_price, mt5_interface)
                    pass # Placeholder for actual management

            for m5_pair_key in config.PRIMARY_MODEL_PAIRS_TIMEFRAMES:
                pair_config = config.HISTORICAL_DATA_SOURCES.get(m5_pair_key)
                if not pair_config: live_logger.warning(f"No config for {m5_pair_key}, skipping."); continue
                
                symbol = pair_config['pair']
                live_logger.debug(f"Processing live for {symbol}...")

                # Load model if not already loaded for this pair
                if m5_pair_key not in live_models_scalers_features:
                    model_file_prefix = f"{symbol}_{pair_config['timeframe_str']}_{config.MODEL_TYPE_FOR_TRADING}"
                    mm = ModelManager(config, live_logger, model_name_prefix=model_file_prefix)
                    model, scaler, f_names = mm.load_model_and_scaler(model_file_prefix)
                    if model and scaler and f_names:
                        live_models_scalers_features[m5_pair_key] = (model, scaler, f_names)
                        live_signal_processors[m5_pair_key] = SignalProcessor(config, live_logger, model, scaler, f_names)
                        live_logger.info(f"Model and SignalProcessor loaded for {symbol}.")
                    else:
                        live_logger.error(f"Failed to load model/scaler for {symbol}. Skipping this pair for now."); continue
                
                model, scaler, feature_names = live_models_scalers_features[m5_pair_key]
                signal_processor = live_signal_processors[m5_pair_key]

                # Fetch live data
                df_h1_live = data_fetcher.fetch_live_candle_data(symbol, config.TIMEFRAME_H1_MT5, num_candles=config.LIVE_NUM_CANDLES_FETCH_H1)
                df_m5_live = data_fetcher.fetch_live_candle_data(symbol, config.TIMEFRAME_M5_MT5, num_candles=config.LIVE_NUM_CANDLES_FETCH_M5)

                if df_m5_live is None or df_m5_live.empty or (df_h1_live is None and config.H1_NUM_SNR_LEVELS > 0) : # H1 optional if no SNR levels
                    live_logger.warning(f"Insufficient live data for {symbol}. Skipping cycle."); continue
                
                # Feature Engineering
                # Adapt FeatureBuilder for live: build features for the latest candle(s)
                feature_builder_live = FeatureBuilder(config, live_logger, market_analyzer)
                # build_features_and_labels should be adapted to take live_h1_snr_levels
                # and process only recent M5 data.
                # For now, conceptual:
                latest_features_df = feature_builder_live.build_features_for_live(df_m5_live, df_h1_live) # Needs new method in FB

                if latest_features_df.empty:
                    live_logger.warning(f"Feature generation failed for latest live data {symbol}."); continue
                
                latest_candle_features_series = latest_features_df.iloc[-1] # Features for the most recent complete M5 candle

                # Check for new entry signals if no open position for this symbol by this bot
                symbol_positions = [p for p in open_positions if p.symbol == symbol]
                if not symbol_positions: # No open trade for this specific symbol by this bot
                    live_logger.info(f"No open trade for {symbol}. Checking for new signals...")
                    candidate_direction = signal_processor.check_candidate_entry_conditions(latest_candle_features_series)
                    if candidate_direction:
                        ml_confirmed_signal = signal_processor.generate_ml_confirmed_signal(latest_candle_features_series, candidate_direction)
                        if ml_confirmed_signal:
                            signal_type = ml_confirmed_signal['signal'] # 'BUY' or 'SELL'
                            live_logger.info(f"ML Confirmed {signal_type} signal for {symbol} at {ml_confirmed_signal['timestamp']}")
                            
                            account_info = mt5_interface.get_account_info()
                            mt5_sym_info = mt5_interface.get_symbol_info(symbol)
                            if not account_info or not mt5_sym_info:
                                live_logger.error(f"Failed to get account/symbol info for {symbol}. Skipping trade."); continue

                            entry_price_ref = latest_candle_features_series[config.CLOSE_COL] # M5 close
                            atr_val = latest_candle_features_series.get(config.M5_ATR_COL_BASE) # M5 ATR
                            if pd.isna(atr_val) or atr_val <= 0:
                                live_logger.warning(f"Invalid ATR for {symbol} ({atr_val}). Cannot calculate SL/TP."); continue

                            sl_price = risk_manager.calculate_stop_loss(entry_price_ref, atr_val, signal_type.lower(), symbol)
                            tp_price = risk_manager.calculate_take_profit(entry_price_ref, sl_price, signal_type.lower(), symbol)
                            
                            trade_volume = risk_manager.get_trade_volume(
                                account_balance=account_info.balance,
                                entry_price=entry_price_ref, # Actual entry will be market
                                sl_price=sl_price,
                                symbol=symbol,
                                symbol_info_mt5=mt5_sym_info # Pass live MT5 info
                            )

                            if trade_volume is None or trade_volume <= 0:
                                live_logger.error(f"Invalid trade volume ({trade_volume}) for {symbol}. Skipping trade."); continue
                            
                            order_type_mt5 = mt5.ORDER_TYPE_BUY if signal_type == 'BUY' else mt5.ORDER_TYPE_SELL
                            trade_comment = f"LiveAI_{signal_type}_{symbol}_{int(time.time())}"
                            
                            live_logger.info(f"Attempting to place {signal_type} order for {symbol}: Vol={trade_volume:.2f}, SL={sl_price:.5f}, TP={tp_price:.5f if tp_price else 'None'}")
                            order_result = mt5_interface.place_market_order(
                                symbol=symbol, order_type_mt5=order_type_mt5, volume=trade_volume,
                                sl_price=sl_price, tp_price=tp_price,
                                magic_number=config.MT5_MAGIC_NUMBER, comment=trade_comment
                            )
                            if order_result and order_result.retcode == mt5.TRADE_RETCODE_DONE:
                                live_logger.info(f"Trade placed successfully for {symbol}: Deal {order_result.deal}, Order {order_result.order}")
                            else:
                                live_logger.error(f"Failed to place trade for {symbol}. Result: {order_result.comment if order_result else 'N/A'}")
                else:
                    live_logger.info(f"Existing trade found for {symbol}. Monitoring, no new entry check.")
            
            data_fetcher.shutdown_mt5() # Close connection if opened by data_fetcher for this cycle
            live_logger.info(f"--- Live Trading Cycle End ---")

    except KeyboardInterrupt:
        live_logger.info("Live trading loop interrupted by user (Ctrl+C).")
    except Exception as e:
        live_logger.critical(f"Critical error in live trading loop: {e}", exc_info=True)
    finally:
        live_logger.info("Shutting down live trading components...")
        if mt5_interface.is_connected: mt5_interface.disconnect()
        data_fetcher.shutdown_mt5() # Ensure DataFetcher's connection is also closed
        app_logger.info("--- Live Trading Mode Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-ML Trading Bot (Multi-Timeframe, Dynamic Strategy)")
    parser.add_argument("--mode", type=str, choices=["train", "backtest", "live"], help="Mode of operation.")
    parser.add_argument("--clear-logs", action="store_true", help="Clear all log files from LOG_DIR.")
    parser.add_argument("--clear-reports", action="store_true", help="Clear all backtest reports from BACKTEST_OUTPUT_DIR.")
    parser.add_argument("--use-smote", action="store_true", help="Enable SMOTE during training (if available).") # From P2
    args = parser.parse_args()

    # Initial setup (logging, directories)
    if not setup_environment():
        app_logger.critical("Failed to setup environment. Exiting.")
        exit(1)

    if args.clear_logs or args.clear_reports:
        app_logger.info("Processing clear options...")
        logging_utils.close_all_file_handlers() # Close handlers before deleting files
        if args.clear_logs: clear_directory_contents(config.LOG_DIR, "Logs")
        if args.clear_reports: clear_directory_contents(config.BACKTEST_OUTPUT_DIR, "Backtest Reports")
        if not args.mode: app_logger.info("Clear options processed. No mode specified, exiting."); exit()

    if args.mode:
        app_logger.info(f"Application started in '{args.mode.upper()}' mode.")
        if args.mode == "train":
            run_training(use_smote_arg=args.use_smote)
        elif args.mode == "backtest":
            run_backtesting()
        elif args.mode == "live":
            run_live_trading()
        else: # Should not be reached due to argparse choices
            app_logger.error(f"Invalid mode: {args.mode}"); parser.print_help()
        app_logger.info(f"Application finished '{args.mode.upper()}' mode.")
    elif not args.clear_logs and not args.clear_reports:
        parser.print_help()
        app_logger.info("No mode or clear action specified. Exiting.")

    logging_utils.close_all_file_handlers() # Ensure all log files are closed at the very end