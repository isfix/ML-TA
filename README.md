# ML-TA

## 1. Project Overview

This project implements an advanced, modular AI-powered trading bot in Python, designed for integration with the MetaTrader 5 (MT5) platform and capable of sophisticated backtesting. It automates the entire workflow from data ingestion and dynamic feature engineering to hyperparameter-tuned machine learning model training (including ensembling), and simulated live execution of trading signals.

The system is engineered to:
- Fetch historical and live price data from MT5 or local CSV files.
- Calculate a wide array of technical indicators.
- Identify **dynamic H1 Support/Resistance (SNR) zones** based on rolling pivot point analysis.
- Construct comprehensive M5 feature vectors incorporating H1 context, M5 indicators, and market dynamics.
- Train Machine Learning models (e.g., RandomForest, XGBoost) using **Optuna for hyperparameter optimization** and optionally create **ensemble models**.
- Save and load models using `joblib` (for model/pipeline objects) and `json` (for feature names).
- Backtest the trading strategy on historical data, simulating **incremental bar-by-bar data fetching from MT5** (Option A2) for realism.
- Implement **flexible risk management** (fixed lot size or dynamic percentage of equity).
- (Framework for) Execute trades on a live MT5 account based on ML-confirmed signals, dynamic SL/TP, and risk rules.

## 2. Core Strategy

1.  **Dynamic H1 SNR Context:**
    - For each H1 candle, the system analyzes a rolling window of recent H1 price data (e.g., last 300 H1 candles).
    - Within this window, it uses a shorter lookback (e.g., 100 H1 candles) to identify significant pivot highs and lows (fractals).
    - These pivots are used to determine multiple dynamic H1 Support and Resistance levels relevant to the current H1 candle.
2.  **M5 Feature Engineering & ML Confirmation:**
    - M5 data is enriched with features derived from M5 technical indicators and the dynamically calculated H1 SNR context (e.g., distance to nearest H1 dynamic support/resistance normalized by M5 ATR).
    - A Machine Learning model (tuned with Optuna, potentially an ensemble) trained on these M5 features provides a probabilistic confirmation for trade entries.
3.  **Entry Rules:**
    - **Candidate Signal:** A rule-based candidate signal is generated (e.g., price near H1 dynamic support + bullish M5 EMA slope for a BUY).
    - **Session Filter:** Candidate signals can be filtered to occur only during specified trading sessions (e.g., London, New York).
    - **ML Confirmation:** The ML model must confirm the candidate signal with a probability exceeding a defined threshold.
4.  **Trade Management:**
    -   **Stop Loss (SL):** Calculated based on M5 Average True Range (ATR) at the time of entry.
    -   **Take Profit (TP):** Primarily targets dynamic H1 SNR levels. If unsuitable, falls back to dynamic M5 pivot levels. If no pivot TP, managed by SL and potential M5 EMA-based exit.
    -   **Risk Management:**
        -   **Flexible Position Sizing:** Configurable to use either a fixed lot size or a dynamic percentage of account equity, risking a certain percentage per trade based on SL distance.
    -   **Breakeven:** SL can be moved to entry + a small buffer after the trade achieves a predefined profit.
    -   **(Optional) Trailing Stop Loss:** Framework for ATR-based trailing SL.

## 3. Key Features

-   **Advanced Multi-Timeframe Analysis**: Dynamic H1 SNR for context, M5 for ML features, entry, and trade management.
-   **Dynamic H1 SNR**: Rolling window pivot analysis for adaptive support and resistance.
-   **Sophisticated Machine Learning Pipeline**:
    -   Hyperparameter optimization using **Optuna**.
    -   Support for **ensemble modeling** (e.g., VotingClassifier).
    -   Models saved using `joblib`, feature names with `json`.
    -   Optional SMOTE for handling class imbalance during training.
-   **Flexible Risk Management**:
    -   Choice between **fixed lot size** or **dynamic percentage-of-equity** position sizing.
    -   ATR-based Stop Loss.
    -   Dynamic Take Profit targeting H1/M5 pivots or EMA-based exit.
    -   Breakeven mechanism.
-   **Data Ingestion**: Supports fetching from MT5 or loading from CSV files for multiple pairs/timeframes.
-   **Comprehensive Feature Engineering (Project 1 Logic)**:
    -   Distances to H1 dynamic S/R levels (normalized by M5 ATR).
    -   M5 EMA distance, spread, and slope features.
    -   Additional H1 context (EMA relationships, distance to H1 EMAs).
    -   M5 candle structure and volume profile features.
    -   Session encoding and cyclical time features.
-   **Realistic Backtesting (Option A2)**:
    -   Simulates incremental, bar-by-bar data fetching from MT5 for the backtest period.
    -   Features and signals are generated dynamically for each simulated candle.
    -   Multi-pair backtesting with consolidated equity curve and performance report.
    -   Detailed reports: trade log, equity curve plot, summary statistics.
-   **Full Live Trading Framework**:
    -   Robust `MT5Interface` for all MT5 interactions.
    -   Main live loop for data fetching, signal processing, and trade execution/management.
    -   Tracks open positions and manages them (placeholder for advanced trailing SL/early exits).
-   **Modular & Configurable**: Well-organized codebase with most parameters in `config.py`.
-   **Extensive Logging**: Detailed logging for all operational modes.
-   **Utility Commands**: CLI options to clear logs and backtest reports.

## 4. Project Structure

(Based on Project 2's structure, with modules adapted from Project 1)
```
ML-TA/
├── main.py                     # Main orchestrator script
├── config.py                   # All configuration variables
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
├── .env                        # For MT5 credentials (optional, gitignored)
├── data/                       # For CSV data files
│   ├── EURUSD_M5.csv
│   └── EURUSD_H1.csv
├── models/                     # For saved ML models (.joblib) and feature names (.json)
│   └── EURUSD_M5_XGBClassifier_ensemble_pipeline.joblib
├── reports/
│   ├── logs/
│   │   ├── trading_bot_app.log
│   │   ├── trading_bot_training.log
│   │   └── ...
│   └── backtests/
│       ├── trade_log.csv
│       ├── equity_curve.png
│       └── performance_report.txt
├── trading_workflow/           # Main package directory (if main.py is outside)
│   ├── __init__.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   └── data_fetcher.py         # Fetches data (MT5/CSV) - P1 logic
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── indicators.py           # TA calculations (class-based) - P1 logic
│   │   ├── market_structure.py     # Dynamic H1 SNR, M5 Pivots - P2 class, new H1 logic
│   │   └── feature_builder.py      # Builds features, defines target - P1 logic in class
│   ├── ml_models/
│   │   ├── __init__.py
│   │   └── model_manager.py        # Optuna, Ensemble, Training, Save/Load - P1 base + new
│   ├── signal_generation/
│   │   ├── __init__.py
│   │   └── signal_processor.py     # Rule-based candidates + ML confirm - P1 logic
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── mt5_interface.py        # MT5 interaction - P1 logic
│   │   └── risk_manager.py         # SL/TP calc, Flexible Position Sizing - P1 base + new
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── backtester.py           # Option A2 backtest engine - P2 report, new sim loop
│   └── utilities/
│       ├── __init__.py
│       ├── logging_utils.py        # Logging setup - P2 logic
│       └── file_utils.py           # File ops (CSV, model saving if not joblib/json) - P2 logic
```
*(Adjust structure slightly if `main.py` and `config.py` are inside a `trading_workflow` directory)*

## 5. Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   MetaTrader 5 Terminal installed and running (for MT5 data source or live trading).
    *   In MT5: `Tools -> Options -> Expert Advisors -> Allow algorithmic trading` AND `Allow DLL imports`.
2.  **Clone/Download Project.**
3.  **Create Virtual Environment (Recommended):**
    *   `python -m venv venv`
    *   Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/macOS).
4.  **Install Dependencies:**
    *   `pip install -r requirements.txt`
5.  **Configure `config.py`:**
    *   **CRUCIAL:** Set MT5 credentials (or use `.env` file), paths, `DATA_SOURCE`.
    *   Define `HISTORICAL_DATA_SOURCES` for all pairs/timeframes, including `backtest_mt5_*` parameters for dynamic risk sizing in backtests.
    *   Set `PRIMARY_MODEL_PAIRS_TIMEFRAMES`.
    *   Adjust strategy parameters (H1 SNR, M5 EMA, ML thresholds, Optuna settings, Risk Management mode, etc.).
6.  **Prepare Data:** If `DATA_SOURCE = "file"`, place CSV files in the `data/` directory. Ensure they match `config.HISTORICAL_DATA_SOURCES` filenames and have correct columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`).
7.  **(Optional) Create `.env` file** in the project root for MT5 credentials if you prefer not to put them directly in `config.py`.

## 6. Running the Workflow (`main.py`)

Ensure MT5 terminal is running if using it as a data source or for live trading.

```bash
python main.py --mode <MODE_NAME> [options]
```

**Modes:**

1.  **`--mode train`**:
    *   Fetches/loads data for pairs in `PRIMARY_MODEL_PAIRS_TIMEFRAMES`.
    *   Performs feature engineering.
    *   Uses Optuna to tune hyperparameters for base models (e.g., XGBoost).
    *   Trains final tuned base models.
    *   Optionally trains an ensemble model.
    *   Saves models (`.joblib`) and feature names (`.json`) to `models/`.
    *   `--use-smote`: (Optional) Apply SMOTE during training if imbalanced-learn is installed.

2.  **`--mode backtest`**:
    *   **Requires models to be trained first.**
    *   Loads the specified model (e.g., ensemble) for each pair.
    *   Simulates trading strategy using **Option A2 (incremental MT5 data fetching)** if `config.BACKTEST_DATA_SOURCE_MT5 = True`, or from files.
    *   Generates features and signals dynamically for each bar.
    *   Applies flexible risk management, dynamic SL/TP, breakeven.
    *   Outputs detailed performance reports (trade log, equity curve, summary) to `reports/backtests/`.

3.  **`--mode live`**:
    *   **Requires models to be trained and MT5 to be running and logged in.**
    *   Connects to MT5 account.
    *   Loads models.
    *   Enters a loop: fetches live data, generates features/signals, manages open trades, and places new trades based on confirmed signals and risk rules.
    *   **WARNING: Live trading involves real financial risk. Test thoroughly on a demo account first.**

**Utility Options:**

*   `--clear-logs`: Clears all log files.
*   `--clear-reports`: Clears all backtest reports.

## 7. Machine Learning Workflow Details

1.  **Data Collection & Preparation (`run_training` in `main.py` -> `DataFetcher`):**
    *   Historical M5 and H1 data fetched/loaded per pair.
2.  **Feature Engineering (`FeatureBuilder`):**
    *   Base TA indicators calculated for M5 and H1 (`Indicators` class).
    *   Dynamic H1 SNR levels calculated (`MarketStructureAnalyzer`).
    *   H1 context merged with M5 data.
    *   Comprehensive M5 features created (Project 1 logic: distances to H1 SNR, M5 EMA features, H1 context, M5 candle/volume, time features).
3.  **Signal Labeling (`FeatureBuilder._define_ml_target`):**
    *   M5 candles labeled (0 for loss, 1 for win) based on future price movement against a hypothetical ATR-based SL/TP (Project 1 logic).
4.  **Model Training & Optimization (`ModelManager`):**
    *   Data split into train/validation/test sets.
    *   **Optuna Hyperparameter Tuning:** For each base model type, Optuna searches for optimal hyperparameters using the validation set.
    *   **Final Model Training:** Base models are trained using the best hyperparameters found by Optuna.
    *   **(Optional) Ensemble Training:** A `VotingClassifier` (or other ensemble method) is trained using the tuned base models.
    *   Models (pipelines including imputer, scaler, classifier) saved as `.joblib`, feature names as `.json`.
5.  **Prediction (Backtest/Live - `SignalProcessor`):**
    *   Appropriate saved model pipeline (e.g., ensemble) and feature names are loaded.
    *   For new data: features generated, aligned with training features, then fed to `pipeline.predict_proba()`.
    *   Probability used by `SignalProcessor` to confirm rule-based candidate signals.

## 8. Troubleshooting

*   **`ModuleNotFoundError`**: Check virtual env, `pip install -r requirements.txt`, `PYTHONPATH`, `__init__.py` files.
*   **MT5 Connection Errors**: Check terminal running, credentials in `config.py` or `.env`, "Allow DLL imports" in MT5.
*   **Data Fetching Issues**: Symbol availability in MT5, correct file paths/formats for CSVs.
*   **Model Loading Errors**: Ensure `train` mode run successfully, paths in `config.MODELS_DIR` are correct.
*   **Feature Mismatches**: Critical. Ensure `FeatureBuilder` consistently produces features in the same order/names as used during training. Saved feature names (`.json`) help align this.
*   **Optuna/SMOTE Issues**: Ensure `optuna` and `imbalanced-learn` are installed.
*   **Backtest Performance/No Trades**: Check rule thresholds, ML confidence, feature quality, labeling, ATR issues, spread. Increase log verbosity.
*   **Live Trading Orders Not Placed**: Check MT5 Journal/Experts tabs. Insufficient funds, invalid volume/SL/TP, market closed, symbol suffixes.

## Disclaimer

Trading financial markets involves substantial risk of loss and is not suitable for all investors. This software is provided "as is" for educational and research purposes. The developers and contributors are not responsible for any financial losses incurred.
