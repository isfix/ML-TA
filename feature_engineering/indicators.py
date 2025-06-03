# feature_engineering/indicators.py
"""
Calculates various technical indicators using pandas_ta or custom implementations.
Based on Project 1's indicators logic, structured as a class.
"""
import pandas as pd
import pandas_ta as ta # Ensure pandas_ta is in requirements.txt
import numpy as np    # For np.nan if needed

# Assuming config and logging_utils are accessible
try:
    import config
    from utilities import logging_utils
except ImportError:
    print("Warning: Could not perform standard config/utilities import in Indicators. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Mock config for standalone testing
    class MockConfig:
        LOG_FILE_APP = "indicators_temp.log"
        HIGH_COL = 'high'; LOW_COL = 'low'; CLOSE_COL = 'close'; VOLUME_COL = 'volume'
        # Add other indicator-specific periods from config if testing them directly
        ATR_PERIOD_INDICATORS = 14 # Example
        M5_EMA_SHORT_PERIOD_ML = 9
        M5_EMA_LONG_PERIOD_ML = 21
        RSI_PERIOD = 14 # Example, P1 used ATR_PERIOD_SR for RSI
        MACD_FAST_PERIOD = 12
        MACD_SLOW_PERIOD = 26
        MACD_SIGNAL_PERIOD = 9
        BBANDS_PERIOD = 20
        BBANDS_STD_DEV = 2


    if 'config' not in locals() and 'config' not in globals():
        config = MockConfig()
        logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)
else: # Standard import successful
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class Indicators:
    """
    A class to encapsulate technical indicator calculation methods.
    Uses pandas_ta where possible.
    """

    def __init__(self):
        """Initializes the Indicators class. Currently no state needed."""
        # self.logger = logger # If each instance needed its own logger
        pass # No instance-specific state needed for static-like methods

    def atr(self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int) -> pd.Series:
        """Calculates Average True Range (ATR)."""
        if not all(isinstance(s, pd.Series) for s in [high_series, low_series, close_series]) or \
           high_series.empty or low_series.empty or close_series.empty:
            logger.warning("ATR: Input series are invalid or empty.")
            return pd.Series(dtype=float, index=close_series.index if hasattr(close_series, 'index') else None)
        if period <= 0: logger.error(f"ATR: Period must be > 0, got {period}."); return pd.Series(np.nan, index=close_series.index)
        try:
            atr_series = ta.atr(high=high_series, low=low_series, close=close_series, length=period)
            if atr_series is None or atr_series.empty: # pandas_ta can return None
                 logger.warning(f"pandas_ta.atr returned None or empty for period {period}.")
                 return pd.Series(np.nan, index=close_series.index)
            return atr_series
        except Exception as e:
            logger.error(f"Error calculating ATR (period {period}) with pandas_ta: {e}", exc_info=True)
            return pd.Series(np.nan, index=close_series.index) # Return NaNs on error

    def ema(self, close_series: pd.Series, period: int) -> pd.Series:
        """Calculates Exponential Moving Average (EMA)."""
        if not isinstance(close_series, pd.Series) or close_series.empty:
            logger.warning("EMA: Input close_series is invalid or empty.")
            return pd.Series(dtype=float, index=close_series.index if hasattr(close_series, 'index') else None)
        if period <= 0: logger.error(f"EMA: Period must be > 0, got {period}."); return pd.Series(np.nan, index=close_series.index)
        try:
            ema_series = ta.ema(close=close_series, length=period)
            if ema_series is None: return pd.Series(np.nan, index=close_series.index)
            return ema_series
        except Exception as e:
            logger.error(f"Error calculating EMA (period {period}) with pandas_ta: {e}", exc_info=True)
            return pd.Series(np.nan, index=close_series.index)

    def rsi(self, close_series: pd.Series, period: int) -> pd.Series:
        """Calculates Relative Strength Index (RSI)."""
        if not isinstance(close_series, pd.Series) or close_series.empty:
            logger.warning("RSI: Input close_series is invalid or empty.")
            return pd.Series(dtype=float, index=close_series.index if hasattr(close_series, 'index') else None)
        if period <= 0: logger.error(f"RSI: Period must be > 0, got {period}."); return pd.Series(np.nan, index=close_series.index)
        try:
            rsi_series = ta.rsi(close=close_series, length=period)
            if rsi_series is None: return pd.Series(np.nan, index=close_series.index)
            return rsi_series
        except Exception as e:
            logger.error(f"Error calculating RSI (period {period}) with pandas_ta: {e}", exc_info=True)
            return pd.Series(np.nan, index=close_series.index)

    def sma(self, close_series: pd.Series, period: int) -> pd.Series:
        """Calculates Simple Moving Average (SMA)."""
        if not isinstance(close_series, pd.Series) or close_series.empty:
            logger.warning("SMA: Input close_series is invalid or empty.")
            return pd.Series(dtype=float, index=close_series.index if hasattr(close_series, 'index') else None)
        if period <= 0: logger.error(f"SMA: Period must be > 0, got {period}."); return pd.Series(np.nan, index=close_series.index)
        try:
            sma_series = ta.sma(close=close_series, length=period)
            if sma_series is None: return pd.Series(np.nan, index=close_series.index)
            return sma_series
        except Exception as e:
            logger.error(f"Error calculating SMA (period {period}) with pandas_ta: {e}", exc_info=True)
            return pd.Series(np.nan, index=close_series.index)

    def macd(self, close_series: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
        """Calculates MACD, MACD Signal, and MACD Histogram."""
        if not isinstance(close_series, pd.Series) or close_series.empty:
            logger.warning("MACD: Input close_series is invalid or empty.")
            return pd.DataFrame(columns=[f"MACD_{fast_period}_{slow_period}_{signal_period}",
                                         f"MACDh_{fast_period}_{slow_period}_{signal_period}",
                                         f"MACDs_{fast_period}_{slow_period}_{signal_period}"], dtype=float)
        if not (fast_period > 0 and slow_period > 0 and signal_period > 0 and fast_period < slow_period):
             logger.error(f"MACD: Invalid periods fast={fast_period}, slow={slow_period}, signal={signal_period}.")
             return pd.DataFrame(np.nan, index=close_series.index, columns=[f"MACD_{fast_period}_{slow_period}_{signal_period}",
                                                                           f"MACDh_{fast_period}_{slow_period}_{signal_period}",
                                                                           f"MACDs_{fast_period}_{slow_period}_{signal_period}"])
        try:
            macd_df = ta.macd(close=close_series, fast=fast_period, slow=slow_period, signal=signal_period)
            if macd_df is None or macd_df.empty:
                logger.warning("pandas_ta.macd returned None or empty.")
                return pd.DataFrame(np.nan, index=close_series.index, columns=[f"MACD_{fast_period}_{slow_period}_{signal_period}",
                                                                           f"MACDh_{fast_period}_{slow_period}_{signal_period}",
                                                                           f"MACDs_{fast_period}_{slow_period}_{signal_period}"])
            # pandas_ta MACD returns columns like MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            # We can return them as is, or rename if a fixed naming convention is desired.
            # For now, returning as is. FeatureBuilder can handle these names.
            return macd_df
        except Exception as e:
            logger.error(f"Error calculating MACD with pandas_ta: {e}", exc_info=True)
            return pd.DataFrame(np.nan, index=close_series.index, columns=[f"MACD_{fast_period}_{slow_period}_{signal_period}",
                                                                           f"MACDh_{fast_period}_{slow_period}_{signal_period}",
                                                                           f"MACDs_{fast_period}_{slow_period}_{signal_period}"])

    def bollinger_bands(self, close_series: pd.Series, period: int, std_dev: float) -> pd.DataFrame:
        """Calculates Bollinger Bands (Upper, Middle, Lower), Bandwidth, PercentB."""
        if not isinstance(close_series, pd.Series) or close_series.empty:
            logger.warning("Bollinger Bands: Input close_series is invalid or empty.")
            # Define expected column names based on pandas_ta output
            bb_cols = [f'BBL_{period}_{std_dev}', f'BBM_{period}_{std_dev}', f'BBU_{period}_{std_dev}',
                       f'BBB_{period}_{std_dev}', f'BBP_{period}_{std_dev}']
            return pd.DataFrame(columns=bb_cols, dtype=float)
        if period <= 0 or std_dev <=0:
            logger.error(f"Bollinger Bands: Invalid period ({period}) or std_dev ({std_dev}).")
            return pd.DataFrame(np.nan, index=close_series.index, columns=bb_cols)
        try:
            bbands_df = ta.bbands(close=close_series, length=period, std=std_dev)
            if bbands_df is None or bbands_df.empty:
                logger.warning("pandas_ta.bbands returned None or empty.")
                return pd.DataFrame(np.nan, index=close_series.index, columns=bb_cols)
            # pandas_ta bbands returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
            # (BBB is Bandwidth, BBP is PercentB)
            return bbands_df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands with pandas_ta: {e}", exc_info=True)
            return pd.DataFrame(np.nan, index=close_series.index, columns=bb_cols)

    # Add other indicators from Project 1's indicators.py if they were custom
    # For example, if P1 had specific EMA cross or spread calculations not covered by FeatureBuilder directly.
    # Project 1's indicators.py used pandas_ta for ATR, RSI, EMA.
    # It had custom logic for EMA cross signal and spread, and volatility (rolling std).
    # FeatureBuilder will handle EMA cross/spread based on the EMAs this class provides.
    # Volatility can also be added here if desired as a standard indicator.

    def volatility(self, close_series: pd.Series, period: int) -> pd.Series:
        """Calculates rolling standard deviation of close prices (volatility)."""
        if not isinstance(close_series, pd.Series) or close_series.empty:
            logger.warning("Volatility: Input close_series is invalid or empty.")
            return pd.Series(dtype=float, index=close_series.index if hasattr(close_series, 'index') else None)
        if period <= 0: logger.error(f"Volatility: Period must be > 0, got {period}."); return pd.Series(np.nan, index=close_series.index)
        try:
            # min_periods can be set to 1 to get output even for shorter windows at the start
            vol_series = close_series.rolling(window=period, min_periods=max(1, period // 2)).std()
            return vol_series
        except Exception as e:
            logger.error(f"Error calculating Volatility (period {period}): {e}", exc_info=True)
            return pd.Series(np.nan, index=close_series.index)