# config.py
"""
Configuration file for the AI-ML Trading Bot.
Merges settings from Project 1 and Project 2, and includes new features.
"""

import os
import logging
import MetaTrader5 as mt5 # For MT5 timeframe constants
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- MT5 Credentials (set here directly, not from .env) ---
USE_DEMO_ACCOUNT = True  # Set to True for demo, False for real

# Demo credentials from .env
MT5_LOGIN_DEMO = int(os.getenv('MT5_LOGIN_DEMO', 0))
MT5_PASSWORD_DEMO = os.getenv('MT5_PASSWORD_DEMO', '')
MT5_SERVER_DEMO = os.getenv('MT5_SERVER_DEMO', '')

# Real credentials from .env
MT5_LOGIN_REAL = int(os.getenv('MT5_LOGIN_REAL', 0))
MT5_PASSWORD_REAL = os.getenv('MT5_PASSWORD_REAL', '')
MT5_SERVER_REAL = os.getenv('MT5_SERVER_REAL', '')

# MT5 Terminal Path from .env
MT5_TERMINAL_PATH = os.getenv('MT5_TERMINAL_PATH', r'C:\Program Files\MetaTrader 5\terminal64.exe')

MT5_LOGIN = MT5_LOGIN_DEMO if USE_DEMO_ACCOUNT else MT5_LOGIN_REAL
MT5_PASSWORD = MT5_PASSWORD_DEMO if USE_DEMO_ACCOUNT else MT5_PASSWORD_REAL
MT5_SERVER = MT5_SERVER_DEMO if USE_DEMO_ACCOUNT else MT5_SERVER_REAL
MT5_PATH = MT5_TERMINAL_PATH
MT5_MAGIC_NUMBER = 123456 # Unique magic number for this bot's trades

# --- General Paths (Project 2 Style) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes config.py is in the project root
PROJECT_ROOT_DIR = BASE_DIR # Corrected: Project root is where config.py resides
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "models") # For .joblib, .json
REPORTS_DIR = os.path.join(PROJECT_ROOT_DIR, "reports")
LOG_DIR = os.path.join(REPORTS_DIR, "logs")
BACKTEST_OUTPUT_DIR = os.path.join(REPORTS_DIR, "backtests")

# Ensure directories exist (can also be handled in main.py's setup_environment)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

# --- Logging Configuration (Project 1 Style Level, Project 2 Style Files) ---
LOG_LEVEL = "DEBUG"  # Corrected: Use string representation e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FILE_APP = os.path.join(LOG_DIR, "trading_bot_app.log")
LOG_FILE_TRAINING = os.path.join(LOG_DIR, "trading_bot_training.log")
LOG_FILE_BACKTEST = os.path.join(LOG_DIR, "trading_bot_backtest.log")
LOG_FILE_LIVE = os.path.join(LOG_DIR, "trading_bot_live.log")
# Note: Project 1 had LOG_FILE_NAME = "trading_bot.log". Project 2 has specific files. We'll use P2 style.

# --- Data Ingestion & Source Configuration ---
# DATA_SOURCE: "mt5" or "file" (Project 1 Style)
DATA_SOURCE = "file" # For training/backtesting. Live will always use "mt5".
START_DATE_HISTORICAL = "2016-01-01"  # Start date for historical data ingestion
END_DATE_HISTORICAL = "2025-05-10"  

# HISTORICAL_DATA_SOURCES: Dictionary mapping keys to data details (Project 2 Style, enhanced for P1 needs)
# This will be the primary way to define data for pairs.
# Add backtest_mt5_* parameters for dynamic risk calculation during backtesting.
HISTORICAL_DATA_SOURCES = {
    "EURUSD_M5": {
        "pair": "EURUSD", "timeframe_str": "M5", "filename": "EURUSD_M5.csv",
        "mt5_timeframe": mt5.TIMEFRAME_M5, "pip_size": 0.0001,
        "backtest_mt5_point": 0.00001, "backtest_mt5_trade_tick_value": 1.0, # Assuming 1 USD per point for 1 lot
        "backtest_mt5_volume_min": 0.01, "backtest_mt5_volume_max": 100.0,
        "backtest_mt5_volume_step": 0.01, "backtest_account_currency": "USD"
    },
    "EURUSD_H1": {
        "pair": "EURUSD", "timeframe_str": "H1", "filename": "EURUSD_H1.csv",
        "mt5_timeframe": mt5.TIMEFRAME_H1, "pip_size": 0.0001
        # H1 doesn't need all backtest_mt5_* params if only used for context
    },
    "GBPUSD_M5": {
        "pair": "GBPUSD", "timeframe_str": "M5", "filename": "GBPUSD_M5.csv",
        "mt5_timeframe": mt5.TIMEFRAME_M5, "pip_size": 0.0001,
        "backtest_mt5_point": 0.00001, "backtest_mt5_trade_tick_value": 1.0,
        "backtest_mt5_volume_min": 0.01, "backtest_mt5_volume_max": 100.0,
        "backtest_mt5_volume_step": 0.01, "backtest_account_currency": "USD"
    },
    "GBPUSD_H1": {
        "pair": "GBPUSD", "timeframe_str": "H1", "filename": "GBPUSD_H1.csv",
        "mt5_timeframe": mt5.TIMEFRAME_H1, "pip_size": 0.0001
    },
    "AUDUSD_M5": {
        "pair": "AUDUSD", "timeframe_str": "M5", "filename": "AUDUSD_M5.csv",
        "mt5_timeframe": mt5.TIMEFRAME_M5, "pip_size": 0.0001,
        "backtest_mt5_point": 0.00001, "backtest_mt5_trade_tick_value": 1.0,
        "backtest_mt5_volume_min": 0.01, "backtest_mt5_volume_max": 100.0,
        "backtest_mt5_volume_step": 0.01, "backtest_account_currency": "USD"
    },
    "GBPJPY_M5": {
        "pair": "GBPJPY", "timeframe_str": "M5", "filename": "GBPJPY_M5.csv",
        "mt5_timeframe": mt5.TIMEFRAME_M5, "pip_size": 0.01,
        "backtest_mt5_point": 0.001, "backtest_mt5_trade_tick_value": 1.0,
        "backtest_mt5_volume_min": 0.01, "backtest_mt5_volume_max": 100.0,
        "backtest_mt5_volume_step": 0.01, "backtest_account_currency": "USD"
    },
    "USDJPY_M5": {
        "pair": "USDJPY", "timeframe_str": "M5", "filename": "USDJPY_M5.csv",
        "mt5_timeframe": mt5.TIMEFRAME_M5, "pip_size": 0.01,
        "backtest_mt5_point": 0.001, "backtest_mt5_trade_tick_value": 1.0,
        "backtest_mt5_volume_min": 0.01, "backtest_mt5_volume_max": 100.0,
        "backtest_mt5_volume_step": 0.01, "backtest_account_currency": "USD"
    },
    # Add other pairs as needed
}
# List of M5 pair keys from HISTORICAL_DATA_SOURCES to be actively trained/traded (Project 2 Style)
PRIMARY_MODEL_PAIRS_TIMEFRAMES = ["EURUSD_M5", "GBPUSD_M5", "AUDUSD_M5", "GBPJPY_M5", "USDJPY_M5"]


# --- Timeframe Constants (Project 1 Style) ---
TIMEFRAME_H1_MT5 = mt5.TIMEFRAME_H1
TIMEFRAME_M5_MT5 = mt5.TIMEFRAME_M5
TIMEFRAME_H1_STR = "H1" # For file naming, suffixes
TIMEFRAME_M5_STR = "M5" # For file naming, suffixes

# --- Session Filtering (UTC times - Project 1 Style) ---
LONDON_SESSION_START = "07:00:00"
LONDON_SESSION_END = "16:00:00"
NY_SESSION_START = "12:00:00"
NY_SESSION_END = "21:00:00"
APPLY_SESSION_FILTER = True # Set to False to disable session filtering
IS_LONDON_SESSION_COL = 'is_london_session' # Added
IS_NY_SESSION_COL = 'is_ny_session'         # Added

# --- Feature Engineering Parameters (Project 1 Style) ---
# H1 Dynamic SNR (Pivots)
H1_SNR_DATA_FETCH_WINDOW = 300  # Number of H1 candles to fetch for context for dynamic SNR
H1_SNR_LOOKBACK_WINDOW = 100    # Number of H1 candles within fetch window to use for pivot calculation
H1_PIVOT_LEFT_STRENGTH = 5
H1_PIVOT_RIGHT_STRENGTH = 5
H1_NUM_SNR_LEVELS = 2           # Number of support and resistance levels to identify (e.g., S1/R1, S2/R2)

# M5 Indicators & Features
ATR_PERIOD_INDICATORS = 14 # General ATR period for indicators if needed
M5_VOLUME_ROLLING_AVG_PERIOD = 20

# M5 EMA features (for ML model, not necessarily for entry rules directly)
M5_EMA_SHORT_PERIOD_ML = 9
M5_EMA_LONG_PERIOD_ML = 21

# H1 EMA features (for ML model context)
H1_EMA_SHORT_PERIOD_ML = 20
H1_EMA_LONG_PERIOD_ML = 50

# M5 Pivots (for Backtester Fallback TP) - Added
M5_PIVOT_LEFT_STRENGTH = 3
M5_PIVOT_RIGHT_STRENGTH = 3
M5_PIVOT_LOOKBACK = 50
M5_PIVOT_SUPPORT_COL = 'm5_pivot_support'
M5_PIVOT_RESISTANCE_COL = 'm5_pivot_resistance'

# M5 Exit EMA (for Backtester) - Added
M5_EXIT_EMA_PERIOD = 12 # Example, adjust as needed
M5_EXIT_EMA_COL = 'm5_exit_ema'


# --- Signal Generation Parameters (Project 1 Style) ---
# Rule-based candidate signal thresholds
H1_SUPPORT_PROXIMITY_THRESHOLD_ATR = 0.75  # Price near H1 Dynamic Support (within X M5 ATRs)
H1_RESISTANCE_PROXIMITY_THRESHOLD_ATR = 0.75 # Price near H1 Dynamic Resistance
M5_EMA_SLOPE_THRESHOLD = 0.005 # Normalized M5 short EMA slope for bullish/bearish confirmation

# ML Model Confirmation
ML_BUY_CONFIRM_THRESHOLD = 0.60  # Min probability from ML model to confirm a rule-based BUY
ML_SELL_CONFIRM_THRESHOLD = 0.60 # Min probability from ML model to confirm a rule-based SELL

# --- Machine Learning Model Configuration ---
# ML_MODEL_TYPES: list of models to train e.g. ['RandomForestClassifier', 'XGBClassifier'] (Project 1 Style)
ML_MODEL_TYPES = ['XGBClassifier'] # Focus on one for now, can be expanded
MODEL_TYPE_FOR_TRADING = 'XGBClassifier_tuned' # Options 'XGBClassifier_ensemble' or 'XGBClassifier_tuned'

# Target variable definition for ML (Project 1 Style)
TARGET_COLUMN_NAME = 'target_ml' # Name of the target column in features_df
TARGET_VARIABLE_LOOKAHEAD_CANDLES_M5 = 12 # e.g., 1 hour on M5
TARGET_TP_RRR = 1.5 # Target R:R for defining winning trades in labeling
TARGET_SL_ATR_MULTIPLIER = 1.0 # ATR multiplier for defining SL in labeling

# Optuna Hyperparameter Optimization
USE_OPTUNA = True
OPTUNA_N_TRIALS = 25  # Number of trials for Optuna per model type per pair
OPTUNA_TIMEOUT_PER_STUDY = 3600 # Seconds, e.g. 1 hour per study
OPTUNA_STUDY_DIRECTION = 'maximize' # 'maximize' or 'minimize' the objective metric
OPTUNA_OBJECTIVE_METRIC = 'f1_weighted' # e.g. 'f1_weighted', 'roc_auc', 'accuracy'

# Ensemble Modeling
TRAIN_ENSEMBLE_MODEL = True
ENSEMBLE_METHOD = 'VotingClassifier' # Currently only VotingClassifier supported
# If VotingClassifier, base estimators will be the best tuned models from Optuna

# SMOTE for imbalanced datasets (can be enabled via command-line arg in main.py)
# USE_SMOTE_DEFAULT = False # Default for training if not specified by CLI

# --- Risk Management Parameters (Flexible - New) ---
RISK_MANAGEMENT_MODE = "DYNAMIC_PERCENTAGE" # "FIXED_LOT" or "DYNAMIC_PERCENTAGE"
FIXED_LOT_SIZE = 0.01  # Used if RISK_MANAGEMENT_MODE is "FIXED_LOT", or as fallback if dynamic fails AND this is > 0

RISK_PER_TRADE_PERCENT = 1.0  # For DYNAMIC_PERCENTAGE: 1.0 means 1% of account balance
MIN_DYNAMIC_LOT_SIZE = 0.01
MAX_DYNAMIC_LOT_SIZE = 2.0

# SL/TP for Execution (Project 1 Style)
DEFAULT_RRR_EXECUTION = 1.5 # Default Reward/Risk Ratio for actual trade execution
SL_ATR_MULTIPLIER_EXECUTION = 1.5
ATR_PERIOD_RISK = 14 # ATR period for SL/TP calculation at execution time

# Breakeven (Project 2 Style, can be integrated with P1 logic)
USE_BREAKEVEN = True
BREAKEVEN_TRIGGER_PIPS = 20 # Pips in profit to trigger breakeven
BREAKEVEN_ADJUST_PIPS = 2    # Pips to set SL beyond entry for breakeven (e.g. entry + 2 pips)

# Trailing Stop Loss (Optional - New Feature)
USE_TRAILING_STOP = False
TRAILING_STOP_ATR_MULTIPLIER = 2.0
TRAILING_STOP_STEP_ATR_MULTIPLIER = 0.5 # How much ATR price has to move further before trailing again

# --- Backtesting Configuration ---
BACKTEST_INITIAL_CAPITAL = 10000.00
BACKTEST_SPREAD_PIPS = 1.0 # Simulated spread in pips for backtesting
# For Option A2 backtesting (simulated live fetching):
BACKTEST_DATA_SOURCE_MT5 = True # If True, backtester uses MT5 for data, else uses files like training

# --- Live Trading Configuration ---
LIVE_TRADING_POLL_INTERVAL_SECONDS = 5 # How often to check for new M5 candle (approx)
LIVE_NUM_CANDLES_FETCH_M5 = 250 # Number of M5 candles to fetch for live indicators
LIVE_NUM_CANDLES_FETCH_H1 = H1_SNR_DATA_FETCH_WINDOW + 50 # Ensure enough for SNR + some buffer

# --- Column Name Constants (Standardized) ---
# These can be used throughout the project to refer to DataFrame columns
# Basic OHLCV
TIMESTAMP_COL = 'Time' # Standard timestamp column for all data
OPEN_COL = 'Open'
HIGH_COL = 'High'
LOW_COL = 'Low'
CLOSE_COL = 'Close'
VOLUME_COL = 'volume' 

# Example feature columns (actual names will be generated by feature_builder)
# M5 ATR for SL/TP calculations (from Project 1's feature_builder)
M5_ATR_COL_BASE = f'atr{TIMEFRAME_M5_STR}' # e.g., 'atr_M5'

# Dynamic H1 SNR Level Columns (names generated based on H1_NUM_SNR_LEVELS)
H1_DYNAMIC_SUPPORT_COLS = [f'h1_dyn_support_{i+1}' for i in range(H1_NUM_SNR_LEVELS)]
H1_DYNAMIC_RESISTANCE_COLS = [f'h1_dyn_resistance_{i+1}' for i in range(H1_NUM_SNR_LEVELS)]
# Distance features to these dynamic levels (example names, actual might vary)
DIST_TO_H1_DYN_SUPPORT_ATR = 'dist_to_h1_dyn_support_atr'
DIST_TO_H1_DYN_RESISTANCE_ATR = 'dist_to_h1_dyn_resistance_atr'

# M5 EMA Slope features (example names)
M5_EMA_SHORT_SLOPE_NORM_COL = f'ema_entry_short{TIMEFRAME_M5_STR}_slope1_norm'

# Pip sizes for symbols (example, RiskManager uses this)
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "AUDUSD": 0.0001,
    "GBPJPY": 0.01,
    # Add other commonly traded pairs if needed as defaults
}