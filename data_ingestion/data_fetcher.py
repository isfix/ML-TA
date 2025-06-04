# data_ingestion/data_fetcher.py
"""
Handles fetching historical and live market data from MetaTrader 5 or local CSV files.
Based on Project 1's DataFetcher.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone 
import time 
import os 

try:
    import config
    from utilities import logging_utils 
except ImportError:
    print("Warning: Could not perform standard config/utilities import in DataFetcher. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    class MockConfig: # Basic mock for fallback
        LOG_FILE_APP = "data_fetcher_temp.log"; DATA_SOURCE = "file"; MT5_PATH = ""; MT5_LOGIN = "testlogin"
        MT5_PASSWORD = "testpassword"; MT5_SERVER = "testserver"; DATA_DIR = "./temp_data/" 
        TIMEFRAME_M5_STR = "M5"; TIMEFRAME_H1_STR = "H1"; TIMEFRAME_M5_MT5 = mt5.TIMEFRAME_M5; TIMEFRAME_H1_MT5 = mt5.TIMEFRAME_H1
        HISTORICAL_DATA_SOURCES = {"EURUSD_M5": {"pair": "EURUSD", "timeframe_str": "M5", "filename": "EURUSD_M5.csv", "mt5_timeframe": mt5.TIMEFRAME_M5},"EURUSD_H1": {"pair": "EURUSD", "timeframe_str": "H1", "filename": "EURUSD_H1.csv", "mt5_timeframe": mt5.TIMEFRAME_H1}}
        TIMESTAMP_COL = 'timestamp'; OPEN_COL = 'open'; HIGH_COL = 'high'; LOW_COL = 'low'; CLOSE_COL = 'close'; VOLUME_COL = 'volume'
    if 'config' not in locals() and 'config' not in globals(): config = MockConfig(); os.makedirs(config.DATA_DIR, exist_ok=True) 
    try: import logging_utils as temp_logging_utils; logger = temp_logging_utils.setup_logger(__name__, config.LOG_FILE_APP) # type: ignore
    except ImportError: logger = logging.getLogger(__name__)
else: 
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class DataFetcher:
    def __init__(self, config_obj, logger_obj): 
        self.config = config_obj
        self.logger = logger_obj
        self.mt5_initialized_by_class = False 

        if self.config.DATA_SOURCE == "mt5": # Or any mode that might use MT5 directly
            self._initialize_mt5()

    def _initialize_mt5(self):
        if mt5.terminal_info() is None: 
            self.logger.info("Attempting to initialize MetaTrader 5...")
            mt5_path_to_use = self.config.MT5_PATH if self.config.MT5_PATH else None
            if not mt5.initialize(login=self.config.MT5_LOGIN, password=self.config.MT5_PASSWORD, server=self.config.MT5_SERVER, path=mt5_path_to_use):
                self.logger.error(f"MT5 initialize failed, error code: {mt5.last_error()}"); return False
            self.mt5_initialized_by_class = True
            self.logger.info(f"MT5 initialized successfully by this DataFetcher instance. Version: {mt5.version()}")
            account_info = mt5.account_info()
            if account_info: self.logger.info(f"Logged in to MT5 account: {account_info.login}, Server: {account_info.server}")
            else: self.logger.warning(f"Could not get account info after MT5 initialize. Error: {mt5.last_error()}.")
        else: self.logger.info("MT5 already initialized.")
        return True

    def shutdown_mt5(self):
        if self.mt5_initialized_by_class and mt5.terminal_info() is not None:
            self.logger.info("Shutting down MetaTrader 5 connection initiated by this DataFetcher instance.")
            mt5.shutdown(); self.mt5_initialized_by_class = False
        # Removed 'elif self.config.DATA_SOURCE == "mt5":' as the flag mt5_initialized_by_class is more direct.

    def _validate_and_format_df(self, df: pd.DataFrame, symbol_for_log: str, source_for_log: str) -> pd.DataFrame | None:
        if df is None or df.empty:
            self.logger.warning(f"No data returned from {source_for_log} for {symbol_for_log}."); return pd.DataFrame() 
        
        df_processed = df.copy()
        timestamp_col_name = self.config.TIMESTAMP_COL
        
        if 'time' in df_processed.columns: 
            df_processed[timestamp_col_name] = pd.to_datetime(df_processed['time'], unit='s', errors='coerce', utc=True)
            if df_processed[timestamp_col_name].isnull().all():
                self.logger.error(f"All values in 'time' column failed to parse as datetime for {symbol_for_log}."); return None
            df_processed.set_index(timestamp_col_name, inplace=True)
            if 'time' != timestamp_col_name: df_processed.drop(columns=['time'], inplace=True, errors='ignore')
        elif timestamp_col_name in df_processed.columns:
            try:
                parsed_dates = pd.to_datetime(df_processed[timestamp_col_name], errors='coerce', utc=True)
                if parsed_dates.isnull().all():
                    self.logger.error(f"All values in column '{timestamp_col_name}' failed to parse as datetime for {symbol_for_log} from {source_for_log}.")
                    return None
                df_processed[timestamp_col_name] = parsed_dates
                df_processed.set_index(timestamp_col_name, inplace=True)
            except Exception as e:
                self.logger.error(f"Error processing column '{timestamp_col_name}' as datetime index for {symbol_for_log} from {source_for_log}: {e}", exc_info=True)
                return None
        elif pd.api.types.is_datetime64_any_dtype(df_processed.index):
            if df_processed.index.tz is None: df_processed.index = df_processed.index.tz_localize('UTC')
            elif df_processed.index.tz != timezone.utc: df_processed.index = df_processed.index.tz_convert('UTC')
        else:
            self.logger.error(f"DataFrame from {source_for_log} for {symbol_for_log} has no 'time' column, no '{timestamp_col_name}' column, and index is not datetime."); return None

        # Drop rows where the index (timestamp) became NaT after coercion
        if df_processed.index.hasnans:
            rows_before_dropna = len(df_processed)
            df_processed.dropna(axis=0, subset=[df_processed.index.name], inplace=True)
            self.logger.warning(f"Dropped {rows_before_dropna - len(df_processed)} rows with NaT timestamps for {symbol_for_log}.")
        if df_processed.empty:
            self.logger.error(f"DataFrame became empty after dropping NaT timestamps for {symbol_for_log}."); return None

        rename_map = {'open': self.config.OPEN_COL, 'high': self.config.HIGH_COL, 'low': self.config.LOW_COL, 'close': self.config.CLOSE_COL, 'tick_volume': self.config.VOLUME_COL, 'volume': self.config.VOLUME_COL, 'real_volume': f"{self.config.VOLUME_COL}_real"}
        actual_rename_map = {k: v for k, v in rename_map.items() if k in df_processed.columns and k !=v}
        df_processed.rename(columns=actual_rename_map, inplace=True)
        
        required_cols = [self.config.OPEN_COL, self.config.HIGH_COL, self.config.LOW_COL, self.config.CLOSE_COL, self.config.VOLUME_COL]
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            self.logger.error(f"DataFrame for {symbol_for_log} is missing required columns after rename: {missing_cols}. Available: {df_processed.columns.tolist()}"); return None
        
        for col in required_cols:
            try: 
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # Coerce errors to NaN for numeric conversion
            except Exception as e: # Broad exception though to_numeric with coerce should handle most
                self.logger.error(f"Unexpected error converting column '{col}' to numeric for {symbol_for_log}: {e}", exc_info=True)
                df_processed[col] = np.nan # Fill with NaN on unexpected error
        
        # Check for NaNs introduced by to_numeric coercion in critical columns
        if df_processed[required_cols].isnull().values.any():
            self.logger.warning(f"NaNs found in OHLCV columns for {symbol_for_log} after numeric conversion. Dropping rows with NaNs in these columns.")
            df_processed.dropna(subset=required_cols, inplace=True)
            if df_processed.empty:
                self.logger.error(f"DataFrame became empty after dropping rows with NaNs in OHLCV columns for {symbol_for_log}."); return None

        self.logger.info(f"Fetched and formatted {len(df_processed)} records for {symbol_for_log} from {source_for_log}.")
        return df_processed[required_cols]

    def fetch_historical_data(self, symbol: str, timeframe_mt5_enum, start_date_dt: datetime, end_date_dt: datetime) -> pd.DataFrame | None:
        self.logger.info(f"Fetching historical data for {symbol} (TF: {timeframe_mt5_enum}) from {start_date_dt} to {end_date_dt} via {self.config.DATA_SOURCE}")
        if self.config.DATA_SOURCE == "mt5":
            if not mt5.terminal_info() and not self._initialize_mt5(): self.logger.error("MT5 not initialized. Cannot fetch MT5 data."); return None
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"Symbol {symbol} not found. Attempting select."); 
                if not mt5.symbol_select(symbol, True): self.logger.warning(f"Failed to select {symbol}. Error: {mt5.last_error()}")
                time.sleep(0.5); symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None: self.logger.error(f"{symbol} still not found. Error: {mt5.last_error()}"); return None
            try: rates = mt5.copy_rates_range(symbol, timeframe_mt5_enum, start_date_dt, end_date_dt)
            except Exception as e: self.logger.error(f"Exception: mt5.copy_rates_range for {symbol}: {e}", exc_info=True); return None
            df_raw_data = pd.DataFrame(rates) if rates is not None and len(rates) > 0 else pd.DataFrame()
            return self._validate_and_format_df(df_raw_data, symbol, "MT5 Historical")
        elif self.config.DATA_SOURCE == "file":
            file_to_load = None
            for key, src_cfg in self.config.HISTORICAL_DATA_SOURCES.items():
                if src_cfg['pair'] == symbol and src_cfg['mt5_timeframe'] == timeframe_mt5_enum: file_to_load = src_cfg.get('filename'); break
            if not file_to_load: self.logger.error(f"No file config for {symbol}, TF {timeframe_mt5_enum}."); return None
            filepath = os.path.join(self.config.DATA_DIR, file_to_load)
            self.logger.info(f"Loading data from file: {filepath}")
            if not os.path.exists(filepath): self.logger.error(f"File not found: {filepath}"); return None
            try:
                df_from_file = pd.read_csv(filepath)
                df_processed = self._validate_and_format_df(df_from_file, symbol, f"File ({file_to_load})")
                if df_processed is None or df_processed.empty: self.logger.error(f"Data validation failed for {filepath}"); return None
                
                # Ensure index is UTC for comparison, _validate_and_format_df should handle this.
                # Just in case, re-verify.
                if df_processed.index.tz is None:
                    df_processed.index = df_processed.index.tz_localize('UTC')
                elif df_processed.index.tz != timezone.utc:
                    df_processed.index = df_processed.index.tz_convert('UTC')
                
                df_filtered = df_processed[(df_processed.index >= start_date_dt) & (df_processed.index <= end_date_dt)]
                if df_filtered.empty:
                     self.logger.warning(f"No data remained for {symbol} after date filtering ({start_date_dt} to {end_date_dt}).")
                return df_filtered
            except Exception as e: self.logger.error(f"Error loading/processing {filepath}: {e}", exc_info=True); return None
        else: self.logger.error(f"Unsupported DATA_SOURCE: {self.config.DATA_SOURCE}"); return None

    def fetch_live_candle_data(self, symbol: str, timeframe_mt5_enum, num_candles: int = 200) -> pd.DataFrame | None:
        self.logger.info(f"Fetching last {num_candles} live candles for {symbol} (TF: {timeframe_mt5_enum})")
        if not mt5.terminal_info() and not self._initialize_mt5(): self.logger.error("MT5 not initialized."); return None
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.warning(f"Symbol {symbol} not found (live). Attempting select.")
            if not mt5.symbol_select(symbol, True): self.logger.warning(f"Failed to select {symbol}. Error: {mt5.last_error()}")
            time.sleep(0.5); symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None: self.logger.error(f"{symbol} still not found. Error: {mt5.last_error()}"); return None
        try: rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5_enum, 0, num_candles)
        except Exception as e: self.logger.error(f"Exception: mt5.copy_rates_from_pos for {symbol}: {e}", exc_info=True); return None
        df_raw_data = pd.DataFrame(rates) if rates is not None and len(rates) > 0 else pd.DataFrame()
        return self._validate_and_format_df(df_raw_data, symbol, "MT5 Live")