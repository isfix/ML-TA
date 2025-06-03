# feature_engineering/market_structure.py
"""
Functions and classes for analyzing market structure.
Implements dynamic H1 Support/Resistance (SNR) level identification based on rolling pivot points.
Also includes M5 Pivot level identification for potential fallback Take Profit targets.
"""
import pandas as pd
import numpy as np

# Assuming config and logging_utils are accessible
try:
    import config
    from utilities import logging_utils
except ImportError:
    print("Warning: Could not perform standard config/utilities import in MarketStructureAnalyzer. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Mock config for standalone testing
    class MockConfig:
        LOG_FILE_APP = "market_structure_temp.log"
        HIGH_COL = 'high'
        LOW_COL = 'low'
        CLOSE_COL = 'close'
        H1_SNR_DATA_FETCH_WINDOW = 300
        H1_SNR_LOOKBACK_WINDOW = 100
        H1_PIVOT_LEFT_STRENGTH = 5
        H1_PIVOT_RIGHT_STRENGTH = 5
        H1_NUM_SNR_LEVELS = 2
        H1_DYNAMIC_SUPPORT_COLS = [f'h1_dyn_support_{i+1}' for i in range(2)]
        H1_DYNAMIC_RESISTANCE_COLS = [f'h1_dyn_resistance_{i+1}' for i in range(2)]
        # M5 Pivot Config
        M5_PIVOT_LEFT_STRENGTH = 3
        M5_PIVOT_RIGHT_STRENGTH = 3
        M5_PIVOT_LOOKBACK = 50
        M5_PIVOT_SUPPORT_COL = 'm5_pivot_support'
        M5_PIVOT_RESISTANCE_COL = 'm5_pivot_resistance'

    if 'config' not in locals() and 'config' not in globals():
        config = MockConfig()
        logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)
else: # Standard import successful
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class MarketStructureAnalyzer:
    def __init__(self, config_obj, logger_obj): # Accept config and logger
        self.config = config_obj
        self.logger = logger_obj
        self.logger.info("MarketStructureAnalyzer initialized.")

    def _find_pivot_points(self, series: pd.Series, left_strength: int, right_strength: int, find_highs=True) -> pd.Series:
        """Identifies pivot points (fractals) in a series."""
        if not isinstance(series, pd.Series) or series.empty:
            self.logger.warning("_find_pivot_points: Input series is invalid or empty.")
            return pd.Series(dtype=bool)
        if left_strength <= 0 or right_strength <= 0:
            self.logger.warning(f"_find_pivot_points: Strengths must be > 0. Got L:{left_strength}, R:{right_strength}.")
            return pd.Series(False, index=series.index)

        n = len(series)
        if n < (left_strength + right_strength + 1):
            self.logger.debug(f"_find_pivot_points: Series length ({n}) too short for L:{left_strength}, R:{right_strength}. No pivots possible.")
            return pd.Series(False, index=series.index)

        is_pivot = pd.Series(False, index=series.index)
        values = series.values # Work with numpy array for potential speedup

        for i in range(left_strength, n - right_strength):
            current_val = values[i]
            is_dominant = True
            # Check left side
            for k in range(1, left_strength + 1):
                if (find_highs and values[i-k] >= current_val) or \
                   (not find_highs and values[i-k] <= current_val):
                    is_dominant = False; break
            if not is_dominant: continue

            # Check right side
            for k in range(1, right_strength + 1):
                if (find_highs and values[i+k] >= current_val) or \
                   (not find_highs and values[i+k] <= current_val):
                    is_dominant = False; break
            if is_dominant:
                is_pivot.iloc[i] = True
        return is_pivot

    def _get_dynamic_snr_for_single_candle(self, h1_lookback_segment: pd.DataFrame, current_h1_ref_price: float) -> dict:
        """
        Calculates dynamic SNR levels for a single H1 candle based on its preceding lookback window.
        h1_lookback_segment: DataFrame of H1_SNR_LOOKBACK_WINDOW candles.
        current_h1_ref_price: Reference price (e.g., close of the current H1 candle) to determine S/R.
        """
        if h1_lookback_segment.empty or not all(col in h1_lookback_segment.columns for col in [self.config.HIGH_COL, self.config.LOW_COL]):
            self.logger.warning("H1 lookback segment is empty or missing H/L columns for dynamic SNR.")
            levels = {}
            for i in range(self.config.H1_NUM_SNR_LEVELS):
                levels[self.config.H1_DYNAMIC_RESISTANCE_COLS[i]] = np.nan
                levels[self.config.H1_DYNAMIC_SUPPORT_COLS[i]] = np.nan
            return levels

        is_pivot_high = self._find_pivot_points(h1_lookback_segment[self.config.HIGH_COL],
                                                self.config.H1_PIVOT_LEFT_STRENGTH,
                                                self.config.H1_PIVOT_RIGHT_STRENGTH, find_highs=True)
        is_pivot_low = self._find_pivot_points(h1_lookback_segment[self.config.LOW_COL],
                                               self.config.H1_PIVOT_LEFT_STRENGTH,
                                               self.config.H1_PIVOT_RIGHT_STRENGTH, find_highs=False)

        pivot_high_prices = h1_lookback_segment.loc[is_pivot_high, self.config.HIGH_COL]
        pivot_low_prices = h1_lookback_segment.loc[is_pivot_low, self.config.LOW_COL]

        # Resistances: pivot highs above current_h1_ref_price, sorted ascending (closest first)
        resistances = sorted(list(set(ph for ph in pivot_high_prices if ph > current_h1_ref_price)))
        # Supports: pivot lows below current_h1_ref_price, sorted descending (closest first)
        supports = sorted(list(set(pl for pl in pivot_low_prices if pl < current_h1_ref_price)), reverse=True)

        snr_levels = {}
        for i in range(self.config.H1_NUM_SNR_LEVELS):
            snr_levels[self.config.H1_DYNAMIC_RESISTANCE_COLS[i]] = resistances[i] if i < len(resistances) else np.nan
            snr_levels[self.config.H1_DYNAMIC_SUPPORT_COLS[i]] = supports[i] if i < len(supports) else np.nan
        
        return snr_levels

    def calculate_all_dynamic_h1_snr(self, full_h1_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates dynamic H1 SNR levels for each candle in the full_h1_df.
        For each H1 candle, it considers the preceding H1_SNR_LOOKBACK_WINDOW candles.
        """
        self.logger.info(f"Calculating all dynamic H1 SNR levels. Total H1 candles: {len(full_h1_df)}")
        if full_h1_df.empty or len(full_h1_df) < self.config.H1_SNR_LOOKBACK_WINDOW:
            self.logger.warning("Not enough H1 data to calculate dynamic SNR.")
            cols = self.config.H1_DYNAMIC_SUPPORT_COLS + self.config.H1_DYNAMIC_RESISTANCE_COLS
            return pd.DataFrame(columns=cols, index=full_h1_df.index if not full_h1_df.empty else None)

        snr_results_list = [] # List to store dicts of SNR levels for each candle

        for i in range(len(full_h1_df)):
            if i < self.config.H1_SNR_LOOKBACK_WINDOW - 1: # Need enough data for the first lookback
                levels = {}
                for k in range(self.config.H1_NUM_SNR_LEVELS): # Populate with NaNs
                    levels[self.config.H1_DYNAMIC_RESISTANCE_COLS[k]] = np.nan
                    levels[self.config.H1_DYNAMIC_SUPPORT_COLS[k]] = np.nan
                snr_results_list.append(levels)
                continue

            # The current H1 candle is at index i. Its features are based on data up to and including itself.
            # The lookback window is from [i - H1_SNR_LOOKBACK_WINDOW + 1] to [i]
            lookback_segment_df = full_h1_df.iloc[i - self.config.H1_SNR_LOOKBACK_WINDOW + 1 : i + 1]
            
            # Use the close of the current H1 candle (end of the lookback segment) as the reference price
            current_h1_ref_price = lookback_segment_df[self.config.CLOSE_COL].iloc[-1]
            if pd.isna(current_h1_ref_price): # Should not happen with clean data
                 current_h1_ref_price = lookback_segment_df[self.config.OPEN_COL].iloc[-1] # Fallback

            snr_levels_for_candle = self._get_dynamic_snr_for_single_candle(lookback_segment_df, current_h1_ref_price)
            snr_results_list.append(snr_levels_for_candle)

        snr_results_df = pd.DataFrame(snr_results_list, index=full_h1_df.index)
        self.logger.info(f"Dynamic H1 SNR calculation complete. Shape: {snr_results_df.shape}")
        return snr_results_df

    def get_m5_pivot_levels(self, m5_df: pd.DataFrame) -> pd.DataFrame:
        """ Identifies dynamic M5 Pivot Support and Resistance levels (Project 2 logic). """
        if not all(col in m5_df.columns for col in [self.config.HIGH_COL, self.config.LOW_COL, self.config.CLOSE_COL]):
            self.logger.error("M5 DataFrame missing H/L/C for pivot detection.")
            return pd.DataFrame(index=m5_df.index, columns=[self.config.M5_PIVOT_SUPPORT_COL, self.config.M5_PIVOT_RESISTANCE_COL])

        left = self.config.M5_PIVOT_LEFT_STRENGTH
        right = self.config.M5_PIVOT_RIGHT_STRENGTH
        lookback = self.config.M5_PIVOT_LOOKBACK
        self.logger.info(f"Identifying M5 Pivots (L:{left},R:{right}) in {lookback}-bar lookback...")

        is_piv_high = self._find_pivot_points(m5_df[self.config.HIGH_COL], left, right, find_highs=True)
        is_piv_low = self._find_pivot_points(m5_df[self.config.LOW_COL], left, right, find_highs=False)

        all_m5_piv_highs = pd.Series(np.nan, index=m5_df.index)
        all_m5_piv_lows = pd.Series(np.nan, index=m5_df.index)
        all_m5_piv_highs.loc[is_piv_high] = m5_df.loc[is_piv_high, self.config.HIGH_COL]
        all_m5_piv_lows.loc[is_piv_low] = m5_df.loc[is_piv_low, self.config.LOW_COL]

        m5_pivots_list = []
        for i in range(len(m5_df)):
            current_candle_data = {}
            window_start_idx = max(0, i - lookback + 1)
            
            window_highs = all_m5_piv_highs.iloc[window_start_idx : i+1].dropna() # Pivots up to current candle
            window_lows = all_m5_piv_lows.iloc[window_start_idx : i+1].dropna()
            
            current_ref_price = m5_df[self.config.CLOSE_COL].iloc[i-1] if i > 0 else m5_df[self.config.OPEN_COL].iloc[i]

            valid_res = window_highs[window_highs >= current_ref_price]
            current_candle_data[self.config.M5_PIVOT_RESISTANCE_COL] = valid_res.min() if not valid_res.empty else np.nan
            
            valid_sup = window_lows[window_lows <= current_ref_price]
            current_candle_data[self.config.M5_PIVOT_SUPPORT_COL] = valid_sup.max() if not valid_sup.empty else np.nan
            m5_pivots_list.append(current_candle_data)
            
        m5_pivots_df = pd.DataFrame(m5_pivots_list, index=m5_df.index)
        self.logger.info(f"M5 Pivot level identification complete. Shape: {m5_pivots_df.shape}")
        return m5_pivots_df