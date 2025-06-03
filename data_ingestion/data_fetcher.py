# data_ingestion/data_fetcher.py
"""
Handles fetching historical and live market data from MetaTrader 5 or local CSV files.
Based on Project 1's DataFetcher.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone # Ensure timezone is imported
import time # For retries
import os # For file operations

# Assuming config and logging_utils are accessible
try:
    import config
    from utilities import logging_utils
except ImportError:
    print("Warning: Could not perform standard config/utilities import in DataFetcher. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Mock config for standalone testing
    class MockConfig:
        LOG_FILE_APP = "data_fetcher_temp.log"
        DATA_SOURCE = "file" # or "mt5"
        MT5_PATH = ""
        MT5_LOGIN = "testlogin"
        MT5_PASSWORD = "testpassword"
        MT5_SERVER = "testserver"
        DATA_DIR = "./temp_data/" # Ensure this exists for testing
        # For HISTORICAL_DATA_SOURCES structure
        TIMEFRAME_M5_STR = "M5"; TIMEFRAME_H1_STR = "H1"
        TIMEFRAME_M5_MT5 = mt5.TIMEFRAME_M5; TIMEFRAME_H1_MT5 = mt5.TIMEFRAME_H1
        HISTORICAL_DATA_SOURCES = {
            "EURUSD_M5": {"pair": "EURUSD", "timeframe_str": "M5", "filename": "EURUSD_M5.csv", "mt5_timeframe": mt5.TIMEFRAME_M5},
            "EURUSD_H1": {"pair": "EURUSD", "timeframe_str": "H1", "filename": "EURUSD_H1.csv", "mt5_timeframe": mt5.TIMEFRAME_H1}
        }
        # Column names
        TIMESTAMP_COL = 'timestamp'; OPEN_COL = 'open'; HIGH_COL = 'high'; LOW_COL = 'low'; CLOSE_COL = 'close'; VOLUME_COL = 'volume'


    if 'config' not in locals() and 'config' not in globals():
        config = MockConfig()
        os.makedirs(config.DATA_DIR, exist_ok=True) # Create temp data dir for testing
        logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)
else: # Standard import successful
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class DataFetcher:
    def __init__(self, config_obj, logger_obj): # Accept config and logger
        self.config = config_obj
        self.logger = logger_obj
        self.mt5_initialized_by_class = False # Flag to track if this instance initialized MT5

        # Auto-initialize MT5 if source is 'mt5' and not already initialized by another DataFetcher instance
        # This is tricky if multiple DataFetcher instances are created.
        # A better approach might be a shared MT5 connection manager or explicit init/shutdown in main.py.
        # For now, following P1's logic: each instance tries to init if needed.
        if self.config.DATA_SOURCE == "mt5":
            self._initialize_mt5()

    def _initialize_mt5(self):
        """Initializes the MetaTrader 5 connection."""
        if mt5.terminal_info() is None: # Check if MT5 is already initialized globally
            self.logger.info("Attempting to initialize MetaTrader 5...")
            mt5_path_to_use = self.config.MT5_PATH if self.config.MT5_PATH else None
            
            if not mt5.initialize(login=self.config.MT5_LOGIN,
                                  password=self.config.MT5_PASSWORD,
                                  server=self.config.MT5_SERVER,
                                  path=mt5_path_to_use):
                self.logger.error(f"MT5 initialize failed, error code: {mt5.last_error()}")
                return False
            
            self.mt5_initialized_by_class = True
            self.logger.info(f"MT5 initialized successfully by this DataFetcher instance. Version: {mt5.version()}")
            
            # Verify login (initialize might handle it, but good to check account_info)
            account_info = mt5.account_info()
            if account_info:
                self.logger.info(f"Logged in to MT5 account: {account_info.login}, Server: {account_info.server}")
            else:
                self.logger.warning(f"Could not get account info after MT5 initialize. Error: {mt5.last_error()}. Ensure login details are correct or terminal is logged in.")
        else:
            self.logger.info("MT5 already initialized (globally or by another instance).")
        return True

    def shutdown_mt5(self):
        """Shuts down the MT5 connection if it was initialized by this specific instance."""
        if self.mt5_initialized_by_class and mt5.terminal_info() is not None:
            self.logger.info("Shutting down MetaTrader 5 connection initiated by this DataFetcher instance.")
            mt5.shutdown()
            self.mt5_initialized_by_class = False
        elif self.config.DATA_SOURCE == "mt5":
            self.logger.debug("MT5 connection was not initiated by this instance or already shut down.")

    def _validate_and_format_df(self, df: pd.DataFrame, symbol_for_log: str, source_for_log: str) -> pd.DataFrame | None:
        """Validates columns, converts 'time' to 'timestamp' index, renames columns."""
        if df is None or df.empty:
            self.logger.warning(f"No data returned from {source_for_log} for {symbol_for_log}.")
            return pd.DataFrame() # Return empty DataFrame for consistency

        df_processed = df.copy()
        if 'time' in df_processed.columns:
            df_processed[self.config.TIMESTAMP_COL] = pd.to_datetime(df_processed['time'], unit='s', utc=True)
            df_processed.set_index(self.config.TIMESTAMP_COL, inplace=True)
        elif self.config.TIMESTAMP_COL in df_processed.columns and not pd.api.types.is_datetime64_any_dtype(df_processed.index):
            # If 'timestamp' is a column but not index, make it so
            df_processed[self.config.TIMESTAMP_COL] = pd.to_datetime(df_processed[self.config.TIMESTAMP_COL], utc=True)
            df_processed.set_index(self.config.TIMESTAMP_COL, inplace=True)
        elif not pd.api.types.is_datetime64_any_dtype(df_processed.index):
            self.logger.error(f"DataFrame from {source_for_log} for {symbol_for_log} has no 'time' column and index is not datetime.")
            return None

        # Standardize column names
        rename_map = {
            'open': self.config.OPEN_COL, 'high': self.config.HIGH_COL,
            'low': self.config.LOW_COL, 'close': self.config.CLOSE_COL,
            'tick_volume': self.config.VOLUME_COL, 'real_volume': f"{self.config.VOLUME_COL}_real"
        }
        # Only rename columns that exist in the DataFrame
        actual_rename_map = {k: v for k, v in rename_map.items() if k in df_processed.columns}
        df_processed.rename(columns=actual_rename_map, inplace=True)

        required_cols = [self.config.OPEN_COL, self.config.HIGH_COL, self.config.LOW_COL, self.config.CLOSE_COL, self.config.VOLUME_COL]
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            self.logger.error(f"DataFrame from {source_for_log} for {symbol_for_log} is missing required columns after rename: {missing_cols}")
            return None
        
        # Ensure numeric types for OHLCV
        for col in required_cols:
            try:
                df_processed[col] = pd.to_numeric(df_processed[col])
            except ValueError as e:
                self.logger.error(f"Could not convert column '{col}' to numeric for {symbol_for_log}: {e}. Dropping NaNs from this column.")
                df_processed.dropna(subset=[col], inplace=True) # Drop rows where this column can't be numeric
                if df_processed.empty: return None # If all rows dropped

        self.logger.info(f"Fetched and formatted {len(df_processed)} records for {symbol_for_log} from {source_for_log}.")
        return df_processed[required_cols]


    def fetch_historical_data(self, symbol: str, timeframe_mt5_enum, start_date_dt: datetime, end_date_dt: datetime) -> pd.DataFrame | None:
        self.logger.info(f"Fetching historical data for {symbol} (TF: {timeframe_mt5_enum}) from {start_date_dt} to {end_date_dt} via {self.config.DATA_SOURCE}")

        if self.config.DATA_SOURCE == "mt5":
            if not mt5.terminal_info() and not self._initialize_mt5():
                 self.logger.error("MT5 not initialized and failed to initialize. Cannot fetch MT5 historical data.")
                 return None
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f"Symbol {symbol} not found by mt5.symbol_info(). Attempting to select.")
                if not mt5.symbol_select(symbol, True):
                    self.logger.warning(f"Failed to select symbol {symbol} in MarketWatch. Error: {mt5.last_error()}")
                time.sleep(0.5) # Wait for MarketWatch
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    self.logger.error(f"Symbol {symbol} still not found. Cannot fetch. Error: {mt5.last_error()}")
                    return None
            
            try:
                rates = mt5.copy_rates_range(symbol, timeframe_mt5_enum, start_date_dt, end_date_dt)
            except Exception as e:
                self.logger.error(f"Exception during mt5.copy_rates_range for {symbol}: {e}", exc_info=True)
                return None
            
            return self._validate_and_format_df(pd.DataFrame(rates) if rates is not None else None, symbol, "MT5 Historical")

        elif self.config.DATA_SOURCE == "file":
            # Find the file config for this symbol and timeframe
            file_to_load = None
            for key, src_cfg in self.config.HISTORICAL_DATA_SOURCES.items():
                if src_cfg['pair'] == symbol and src_cfg['mt5_timeframe'] == timeframe_mt5_enum:
                    file_to_load = src_cfg.get('filename')
                    break
            
            if not file_to_load:
                self.logger.error(f"No file configuration found for {symbol} and timeframe {timeframe_mt5_enum} in HISTORICAL_DATA_SOURCES.")
                return None
            
            filepath = os.path.join(self.config.DATA_DIR, file_to_load)
            self.logger.info(f"Loading data from file: {filepath}")
            if not os.path.exists(filepath):
                self.logger.error(f"File not found: {filepath}")
                return None
            try:
                # Assuming CSV has 'timestamp' column or index is timestamp
                df_from_file = pd.read_csv(filepath, parse_dates=[self.config.TIMESTAMP_COL], index_col=self.config.TIMESTAMP_COL)
                # Filter by date range
                df_filtered = df_from_file[(df_from_file.index >= start_date_dt) & (df_from_file.index <= end_date_dt)]
                return self._validate_and_format_df(df_filtered.reset_index(), symbol, f"File ({file_to_load})") # reset_index because _validate_and_format_df expects 'timestamp' col or sets index
            except Exception as e:
                self.logger.error(f"Error loading or processing data from {filepath}: {e}", exc_info=True)
                return None
        else:
            self.logger.error(f"Unsupported DATA_SOURCE: {self.config.DATA_SOURCE}")
            return None

    def fetch_live_candle_data(self, symbol: str, timeframe_mt5_enum, num_candles: int = 200) -> pd.DataFrame | None:
        self.logger.info(f"Fetching last {num_candles} live candles for {symbol} (TF: {timeframe_mt5_enum})")
        # Live data always comes from MT5, regardless of config.DATA_SOURCE for historical
        if not mt5.terminal_info() and not self._initialize_mt5():
             self.logger.error("MT5 not initialized. Cannot fetch live data.")
             return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.warning(f"Symbol {symbol} not found (live). Attempting select.")
            if not mt5.symbol_select(symbol, True): self.logger.warning(f"Failed to select {symbol}. Error: {mt5.last_error()}")
            time.sleep(0.5); symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None: self.logger.error(f"{symbol} still not found. Error: {mt5.last_error()}"); return None
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5_enum, 0, num_candles)
        except Exception as e:
            self.logger.error(f"Exception during mt5.copy_rates_from_pos for {symbol}: {e}", exc_info=True)
            return None
        
        return self._validate_and_format_df(pd.DataFrame(rates) if rates is not None else None, symbol, "MT5 Live")