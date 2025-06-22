# main.py
"""
Main entry point for the AI-ML Trading Bot.
Handles different modes of operation: train, backtest, live.
Implements a multi-timeframe strategy with dynamic H1 SNR, M5 ML features & entry.
ML models are tuned with Optuna and can be ensembled.
Risk management is flexible (fixed lot or dynamic percentage).
Backtesting simulates live data fetching from MT5.
"""
import sys 
import os  
import argparse
import pandas as pd
import numpy as np
import time
import shutil
import logging 
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5

# Ensure the project root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
except ImportError as e:
    print(f"FATAL ERROR: Could not import 'config.py'. \nProject root: {PROJECT_ROOT}\nSys.path: {sys.path}\nError: {e}")
    exit(1)
except Exception as e: 
    print(f"FATAL ERROR: Unexpected error importing 'config.py': {e}")
    exit(1)

from utilities import logging_utils, file_utils
from data_ingestion.data_fetcher import DataFetcher
from feature_engineering.market_structure import MarketStructureAnalyzer
from feature_engineering.feature_builder import FeatureBuilder 
from ml_models.model_manager import ModelManager
from signal_generation.signal_processor import SignalProcessor
from execution.mt5_interface import MT5Interface
from execution.risk_manager import RiskManager
from backtesting.backtester import Backtester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger("MainApp")


def setup_environment():
    global app_logger 
    app_logger = logging_utils.setup_logger(
        'MainApp',
        config.LOG_FILE_APP,
        level_str=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'INFO'
    )
    app_logger.info("Environment setup: Logging initialized.")

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
    return True

def clear_directory_contents(directory_path, dir_name_for_log):
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
    data_fetcher = DataFetcher(config, training_logger) 

    if not config.PRIMARY_MODEL_PAIRS_TIMEFRAMES:
        training_logger.error("No M5 pairs in config.PRIMARY_MODEL_PAIRS_TIMEFRAMES. Aborting training.")
        return

    for m5_pair_key in config.PRIMARY_MODEL_PAIRS_TIMEFRAMES:
        training_logger.info(f"===== Training for M5 Pair: {m5_pair_key} =====")
        m5_pair_config = config.HISTORICAL_DATA_SOURCES.get(m5_pair_key)
        if not m5_pair_config or m5_pair_config.get('timeframe_str', '').upper() != 'M5':
            training_logger.error(f"Config for M5 pair key '{m5_pair_key}' is invalid or not found. Skipping."); continue

        pair_name = m5_pair_config['pair']
        h1_pair_key = f"{pair_name}_{config.TIMEFRAME_H1_STR}" 
        h1_pair_config = config.HISTORICAL_DATA_SOURCES.get(h1_pair_key)

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
        
        if config.DATA_SOURCE == "mt5": 
            data_fetcher.shutdown_mt5() 

        try:
            market_analyzer = MarketStructureAnalyzer(config, training_logger) 
            feature_builder = FeatureBuilder(config, training_logger, market_analyzer)

            training_logger.info("Building features and labels...")
            features_df_labeled = feature_builder.build_features_and_labels(df_m5_raw, df_h1_raw)

        except Exception as e:
            training_logger.error(f"Error during feature engineering for {pair_name}: {e}", exc_info=True); continue

        if features_df_labeled is None or features_df_labeled.empty or config.TARGET_COLUMN_NAME not in features_df_labeled.columns:
            training_logger.error(f"Feature or target column '{config.TARGET_COLUMN_NAME}' not generated for {pair_name}. Skipping."); continue
        
        model_manager = ModelManager(config, training_logger, model_name_prefix=f"{pair_name}_{m5_pair_config['timeframe_str']}", use_smote_cli=use_smote_arg)
        
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_from_prep = model_manager.prepare_data_for_model(features_df_labeled)
        if X_train is None or X_train.empty: 
            training_logger.error(f"Data preparation failed for {pair_name}. Skipping model training."); continue

        trained_model_pipelines = [] 
        for model_type_name in config.ML_MODEL_TYPES:
            training_logger.info(f"--- Optimizing and Training {model_type_name} for {pair_name} ---")
            best_params = None
            if config.USE_OPTUNA:
                training_logger.info(f"Running Optuna for {model_type_name}...")
                best_params, best_score = model_manager.optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type_name)
                if best_params:
                    training_logger.info(f"Optuna best params for {model_type_name}: {best_params}, Best score: {best_score:.4f}")
                else:
                    training_logger.warning(f"Optuna did not yield best_params for {model_type_name}. Using defaults.")

            training_logger.info(f"Training final {model_type_name} with {'best found' if best_params else 'default'} params...")
            final_pipeline = model_manager.train_individual_model(
                X_train, y_train, X_val, y_val, 
                model_type_name,
                model_params_override=best_params 
            )

            if final_pipeline:
                model_filename_prefix_individual = f"{pair_name}_{m5_pair_config['timeframe_str']}_{model_type_name}{'_tuned' if best_params else ''}"
                model_manager.save_model_pipeline(final_pipeline, model_filename_prefix_individual)
                trained_model_pipelines.append({'name': model_type_name, 'pipeline': final_pipeline, 'params': best_params or 'default'})
        
        if not trained_model_pipelines:
            training_logger.error(f"No base models trained successfully for {pair_name}."); continue

        if config.TRAIN_ENSEMBLE_MODEL and len(trained_model_pipelines) >= 1: 
            training_logger.info(f"--- Training Ensemble Model for {pair_name} ---")
            ensemble_pipeline = model_manager.train_ensemble(X_train, y_train, X_val, y_val, trained_model_pipelines)
            if ensemble_pipeline:
                ensemble_filename_prefix = f"{pair_name}_{m5_pair_config['timeframe_str']}_{config.ENSEMBLE_METHOD if hasattr(config, 'ENSEMBLE_METHOD') else 'ensemble'}"
                model_manager.save_model_pipeline(ensemble_pipeline, ensemble_filename_prefix)
                training_logger.info(f"Ensemble model for {pair_name} trained and saved.")
            else:
                training_logger.warning(f"Failed to train ensemble model for {pair_name}.")
        
        training_logger.info(f"===== Finished training for M5 Pair: {m5_pair_key} =====")
    app_logger.info("--- Training Mode Finished ---")


# In main.py
def run_backtesting():
    app_logger.info("--- Starting Backtesting Mode (Pre-computed Features & Signals) ---")
    backtest_logger = logging_utils.setup_logger('BacktestProcess', config.LOG_FILE_BACKTEST, level_str=config.LOG_LEVEL)
    
    data_fetcher = DataFetcher(config, backtest_logger)
    market_analyzer = MarketStructureAnalyzer(config, backtest_logger)
    # FeatureBuilder is instantiated once to process all data for a pair
    feature_builder_instance = FeatureBuilder(config, backtest_logger, market_analyzer)
    risk_manager_instance = RiskManager(config, backtest_logger)

    all_featured_data_for_backtest = {} # Dict: m5_pair_key -> DataFrame with ALL features
    all_final_signals_for_backtest = {} # Dict: m5_pair_key -> Series of final signals

    for m5_pair_key in config.PRIMARY_MODEL_PAIRS_TIMEFRAMES:
        backtest_logger.info(f"===== Processing data for Backtest: {m5_pair_key} =====")
        m5_pair_config = config.HISTORICAL_DATA_SOURCES.get(m5_pair_key)
        if not m5_pair_config: backtest_logger.error(f"No config for {m5_pair_key}. Skipping."); continue
        
        pair_name = m5_pair_config['pair']
        h1_pair_key = f"{pair_name}_{config.TIMEFRAME_H1_STR}"
        h1_pair_config = config.HISTORICAL_DATA_SOURCES.get(h1_pair_key)

        # 1. Fetch all historical data for the pair
        try:
            start_dt = datetime.fromisoformat(config.START_DATE_HISTORICAL).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(config.END_DATE_HISTORICAL).replace(tzinfo=timezone.utc)
        except ValueError as e:
            backtest_logger.error(f"Invalid date format: {e}. Skipping {m5_pair_key}."); continue

        df_m5_raw = data_fetcher.fetch_historical_data(pair_name, m5_pair_config['mt5_timeframe'], start_dt, end_dt)
        if df_m5_raw is None or df_m5_raw.empty:
            backtest_logger.error(f"No M5 data for {pair_name}. Skipping."); continue
        
        df_h1_raw = None
        if h1_pair_config:
            df_h1_raw = data_fetcher.fetch_historical_data(pair_name, h1_pair_config['mt5_timeframe'], start_dt, end_dt)
            if df_h1_raw is None or df_h1_raw.empty: backtest_logger.warning(f"No H1 data for {pair_name}. H1 features will be limited.")
        
        # 2. Build ALL features (including backtester-specific ones like M5 pivots, exit EMA)
        # The `build_features_and_labels` method creates ML features AND backtester features, but we don't need the 'target_ml' column for backtest execution.
        # We can call it and then drop the target, or have a separate method in FeatureBuilder.
        # Let's assume `build_features_and_labels` returns everything, and we just use what's needed.
        # Or better: `feature_builder_instance.build_features_for_live(df_m5_raw, df_h1_raw)` if it includes backtester cols.
        # For now, let's assume `build_features_and_labels` is used and we ignore the target.
        
        # We need a method in FeatureBuilder that builds all necessary features *without* the ML target label for backtesting.
        # Let's assume FeatureBuilder has `build_all_features_for_trading(df_m5_raw, df_h1_raw)`
        # This method would call: _add_base_ta_indicators, _align_and_merge_h1_context, _create_final_ml_features, _add_backtester_specific_features
        
        try:
            # This method should build all indicators, H1 context, ML features, M5 pivots, M5 exit EMA
            # It should NOT do ML target labeling.
            featured_data_for_pair = feature_builder_instance.build_features_for_live(df_m5_raw, df_h1_raw) # NEW METHOD NEEDED IN FeatureBuilder
        except Exception as e:
            backtest_logger.error(f"Error building features for {pair_name} backtest: {e}", exc_info=True); continue
            
        if featured_data_for_pair.empty:
            backtest_logger.error(f"Feature generation resulted in empty DataFrame for {pair_name}. Skipping."); continue
        
        all_featured_data_for_backtest[m5_pair_key] = featured_data_for_pair

        # 3. Load ML Model and Generate All Signals
        model_file_prefix = f"{pair_name}_{m5_pair_config['timeframe_str']}_{config.MODEL_TYPE_FOR_TRADING}"
        model_manager = ModelManager(config, backtest_logger, model_name_prefix=model_file_prefix)
        pipeline, feature_names = model_manager.load_model_pipeline(model_file_prefix) # Using new method name

        if not pipeline or not feature_names:
            backtest_logger.error(f"Failed to load model/pipeline for {pair_name}. Skipping."); continue
        
        # Ensure all required ML features are present in featured_data_for_pair
        ml_features_for_pred_df = featured_data_for_pair[feature_names] # Select only ML features in correct order

        # Get probabilities for all candles
        # predict_proba returns shape (n_samples, n_classes)
        # We need to process these probabilities with SignalProcessor's logic
        # This is slightly different from P1's SignalProcessor which took one candle.
        # We need a batch version or iterate.
        # For simplicity, let's assume SignalProcessor can be adapted or we iterate.

        signal_processor = SignalProcessor(config, backtest_logger, pipeline, None, feature_names)
        
        # Iterate through the featured_data_for_pair to generate signals row by row
        # This is less efficient than batch but matches SignalProcessor's current design
        signals_list = []
        for timestamp, row_features in ml_features_for_pred_df.iterrows(): # Iterate over ML features part
            # SignalProcessor needs the full feature row (including non-ML features for rules)
            full_feature_row_for_signal = featured_data_for_pair.loc[timestamp]
            
            candidate_dir = signal_processor.check_candidate_entry_conditions(full_feature_row_for_signal)
            final_signal_val = 0
            if candidate_dir:
                ml_confirmed = signal_processor.generate_ml_confirmed_signal(full_feature_row_for_signal, candidate_dir)
                if ml_confirmed:
                    final_signal_val = 1 if ml_confirmed['signal'] == 'BUY' else -1
            signals_list.append(final_signal_val)
        
        all_final_signals_for_backtest[m5_pair_key] = pd.Series(signals_list, index=ml_features_for_pred_df.index)
        backtest_logger.info(f"Generated {len(signals_list)} signals for {pair_name}. Non-zero: {(all_final_signals_for_backtest[m5_pair_key] != 0).sum()}")

    data_fetcher.shutdown_mt5() # Close MT5 if it was used

    if not all_featured_data_for_backtest or not all_final_signals_for_backtest:
        backtest_logger.error("No featured data or signals prepared for any pair. Aborting backtest."); return

    # 4. Run Backtester
    backtester_engine = Backtester(
        config_obj=config,
        logger_obj=backtest_logger,
        initial_capital=config.BACKTEST_INITIAL_CAPITAL,
        all_featured_data=all_featured_data_for_backtest,
        all_final_signals=all_final_signals_for_backtest,
        risk_manager_instance=risk_manager_instance
    )
    
    backtest_logger.info("Running backtest simulation with pre-computed features/signals...")
    try:
        performance_metrics, trade_log_df, equity_curve_df = backtester_engine.run_backtest()
        # Reporting is handled by backtester.generate_report()
    except Exception as e:
        backtest_logger.error(f"Critical error during backtest execution: {e}", exc_info=True)

    app_logger.info("--- Backtesting Mode Finished ---")

def run_live_trading():
    app_logger.info("--- Starting Live Trading Mode ---")
    # Only log to file, not to console
    live_logger = logging_utils.setup_logger('LiveTradingProcess', config.LOG_FILE_LIVE, level_str=config.LOG_LEVEL, console_output=False)
    print("Live Trading Running...")
    mt5_interface = MT5Interface(config, live_logger)
    if not mt5_interface.is_connected: 
        live_logger.error("Failed to connect to MT5. Live trading cannot start."); return

    data_fetcher = DataFetcher(config, live_logger) 
    risk_manager = RiskManager(config, live_logger)
    market_analyzer = MarketStructureAnalyzer(config, live_logger)
    
    live_models_scalers_features = {} 
    live_signal_processors = {}      
    last_h1_candle_time = None
    snr_cache = {}
    last_signal_time = {}
    poll_interval_seconds = 1  # Fetch every second

    try:
        while True:
            now_utc = datetime.now(timezone.utc)
            for m5_pair_key in config.PRIMARY_MODEL_PAIRS_TIMEFRAMES:
                pair_config_details = config.HISTORICAL_DATA_SOURCES.get(m5_pair_key)
                if not pair_config_details: live_logger.warning(f"No config for {m5_pair_key}, skipping."); continue
                symbol = pair_config_details['pair']
                live_logger.debug(f"Processing live for {symbol}...")

                if m5_pair_key not in live_models_scalers_features:
                    model_type_suffix = getattr(config, 'MODEL_TYPE_FOR_TRADING', 'ensemble')
                    model_file_prefix = f"{symbol}_{pair_config_details['timeframe_str']}_{model_type_suffix}"
                    mm = ModelManager(config, live_logger)
                    pipeline, f_names = mm.load_model_pipeline(model_file_prefix)
                    if pipeline and f_names:
                        live_models_scalers_features[m5_pair_key] = (pipeline, f_names) 
                        live_signal_processors[m5_pair_key] = SignalProcessor(config, live_logger, pipeline, None, f_names)
                        live_logger.info(f"Model and SignalProcessor loaded for {symbol}.")
                    else:
                        live_logger.error(f"Failed to load model/pipeline for {symbol}. Skipping this pair for now."); continue
                loaded_pipeline, loaded_feature_names = live_models_scalers_features[m5_pair_key]
                signal_processor = live_signal_processors[m5_pair_key]

                # Fetch latest H1 and M5 data
                df_h1_live = data_fetcher.fetch_live_candle_data(symbol, config.TIMEFRAME_H1_MT5, num_candles=config.LIVE_NUM_CANDLES_FETCH_H1)
                df_m5_live = data_fetcher.fetch_live_candle_data(symbol, config.TIMEFRAME_M5_MT5, num_candles=config.LIVE_NUM_CANDLES_FETCH_M5)
                if df_m5_live is None or df_m5_live.empty or \
                   ((df_h1_live is None or df_h1_live.empty) and config.H1_NUM_SNR_LEVELS > 0) : 
                    live_logger.warning(f"Insufficient live data for {symbol}. Skipping cycle."); continue

                # Update SNR only if new H1 candle
                h1_last_candle_time = df_h1_live.index[-1] if not df_h1_live.empty else None
                if m5_pair_key not in snr_cache or last_h1_candle_time != h1_last_candle_time:
                    snr_cache[m5_pair_key] = market_analyzer.calculate_all_dynamic_h1_snr(df_h1_live)
                    last_h1_candle_time = h1_last_candle_time
                    live_logger.info(f"SNR updated for {symbol} at {h1_last_candle_time}")

                feature_builder_live = FeatureBuilder(config, live_logger, market_analyzer)
                latest_features_df = feature_builder_live.build_features_for_live(df_m5_live, df_h1_live)
                if latest_features_df is None or latest_features_df.empty: 
                    live_logger.warning(f"Feature generation failed for latest live data {symbol}."); continue
                latest_candle_features_series = latest_features_df.iloc[-1] 

                open_positions = mt5_interface.get_open_positions(magic_number=config.MT5_MAGIC_NUMBER)
                symbol_positions = [p for p in open_positions if p.symbol == symbol]
                # Check for SNR interaction (implement your SNR interaction logic here)
                snr_levels = snr_cache.get(m5_pair_key)
                price = latest_candle_features_series[config.CLOSE_COL]
                interacted = False
                if snr_levels is not None:
                    # Support dict, Series, or ndarray
                    if hasattr(snr_levels, 'values') and callable(getattr(snr_levels, 'values', None)):
                        snr_iter = snr_levels.values()
                    elif hasattr(snr_levels, 'values'):
                        snr_iter = snr_levels.values
                    else:
                        snr_iter = snr_levels
                    for snr in snr_iter:
                        # If snr is array-like, check all elements
                        if isinstance(snr, (np.ndarray, pd.Series, list)):
                            if np.any(np.abs(price - np.array(snr)) < config.SNR_INTERACTION_THRESHOLD):
                                interacted = True
                                break
                        else:
                            if abs(price - snr) < config.SNR_INTERACTION_THRESHOLD:
                                interacted = True
                                break
                if interacted:
                    # Only analyze once per interaction (per symbol)
                    last_sig = last_signal_time.get(symbol)
                    if not last_sig or (now_utc - last_sig).total_seconds() > poll_interval_seconds:
                        if not symbol_positions: 
                            live_logger.info(f"No open trade for {symbol}. Checking for new signals...")
                            candidate_direction = signal_processor.check_candidate_entry_conditions(latest_candle_features_series)
                            if candidate_direction:
                                ml_confirmed_signal = signal_processor.generate_ml_confirmed_signal(latest_candle_features_series, candidate_direction)
                                if ml_confirmed_signal:
                                    signal_type = ml_confirmed_signal['signal'] 
                                    live_logger.info(f"ML Confirmed {signal_type} signal for {symbol} at {ml_confirmed_signal['timestamp']}")
                                    account_info = mt5_interface.get_account_info()
                                    mt5_sym_info = mt5_interface.get_symbol_info(symbol)
                                    if not account_info or not mt5_sym_info:
                                        live_logger.error(f"Failed to get account/symbol info for {symbol}. Skipping trade."); continue
                                    entry_price_ref = latest_candle_features_series[config.CLOSE_COL] 
                                    atr_val = latest_candle_features_series.get(config.M5_ATR_COL_BASE) 
                                    if pd.isna(atr_val) or atr_val <= 0:
                                        live_logger.warning(f"Invalid ATR for {symbol} ({atr_val}). Cannot calculate SL/TP."); continue
                                    sl_price = risk_manager.calculate_stop_loss(entry_price_ref, atr_val, signal_type.lower(), symbol)
                                    tp_price = risk_manager.calculate_take_profit(entry_price_ref, sl_price, signal_type.lower(), symbol)
                                    trade_volume = risk_manager.get_trade_volume(
                                        account_balance=account_info.balance,
                                        entry_price=entry_price_ref, 
                                        sl_price=sl_price,
                                        symbol=symbol,
                                        symbol_info_mt5=mt5_sym_info 
                                    )
                                    if trade_volume is None or trade_volume <= 0:
                                        live_logger.error(f"Invalid trade volume ({trade_volume}) for {symbol}. Skipping trade."); continue
                                    order_type_mt5_val = mt5.ORDER_TYPE_BUY if signal_type == 'BUY' else mt5.ORDER_TYPE_SELL
                                    trade_comment = f"LiveAI_{signal_type}_{symbol}_{int(time.time())}"
                                    live_logger.info(f"Attempting to place {signal_type} order for {symbol}: Vol={trade_volume:.2f}, SL={sl_price:.5f}, TP={tp_price if tp_price else 'None'}")
                                    order_result = mt5_interface.place_market_order(
                                        symbol=symbol, order_type_mt5=order_type_mt5_val, volume=trade_volume,
                                        sl_price=sl_price, tp_price=tp_price,
                                        magic_number=config.MT5_MAGIC_NUMBER, comment=trade_comment
                                    )
                                    if order_result and order_result.retcode == mt5.TRADE_RETCODE_DONE:
                                        live_logger.info(f"Trade placed successfully for {symbol}: Deal {order_result.deal}, Order {order_result.order}")
                                    else:
                                        live_logger.error(f"Failed to place trade for {symbol}. Result: {order_result.comment if order_result else 'N/A'}")
                                    last_signal_time[symbol] = now_utc
                        else:
                            live_logger.info(f"Existing trade found for {symbol}. Monitoring, no new entry check.")
            time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        live_logger.info("Live trading loop interrupted by user (Ctrl+C).")
    except Exception as e:
        live_logger.critical(f"Critical error in live trading loop: {e}", exc_info=True)
    finally:
        live_logger.info("Shutting down live trading components...")
        if hasattr(mt5_interface, 'is_connected') and mt5_interface.is_connected: 
            mt5_interface.disconnect()
        if hasattr(data_fetcher, 'mt5_initialized_by_class') and data_fetcher.mt5_initialized_by_class:
             data_fetcher.shutdown_mt5() 
        app_logger.info("--- Live Trading Mode Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-ML Trading Bot (Multi-Timeframe, Dynamic Strategy)")
    parser.add_argument("--mode", type=str, choices=["train", "backtest", "live"], help="Mode of operation.")
    parser.add_argument("--clear-logs", action="store_true", help="Clear all log files from LOG_DIR.")
    parser.add_argument("--clear-reports", action="store_true", help="Clear all backtest reports from BACKTEST_OUTPUT_DIR.")
    parser.add_argument("--use-smote", action="store_true", help="Enable SMOTE during training (if available).")
    args = parser.parse_args()

    if not setup_environment():
        logging.critical("Failed to setup environment. Exiting.") 
        exit(1)

    if args.clear_logs or args.clear_reports:
        app_logger.info("Processing clear options...")
        logging_utils.close_all_file_handlers() 
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
        else: 
            app_logger.error(f"Invalid mode: {args.mode}"); parser.print_help()
        app_logger.info(f"Application finished '{args.mode.upper()}' mode.")
    elif not args.clear_logs and not args.clear_reports: 
        parser.print_help()
        app_logger.info("No mode or clear action specified. Exiting.")

    logging_utils.close_all_file_handlers()