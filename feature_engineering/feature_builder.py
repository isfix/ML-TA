# feature_engineering/feature_builder.py
"""
Builds features for the ML model by combining M5 data with dynamic H1 SNR context,
M5 technical indicators, M5 ATR. Defines ML target.
Also adds M5 pivot levels and M5 exit EMA for the backtester's use.
Ensures all output columns are created, even if with NaNs.
"""
import pandas as pd
import numpy as np
from datetime import time as dtime # For session checking

# Assuming config, logging_utils, Indicators, MarketStructureAnalyzer are accessible
try:
    import config
    from utilities import logging_utils
    from .indicators import Indicators
    from .market_structure import MarketStructureAnalyzer
except ImportError:
    print("FATAL: Could not perform standard imports in FeatureBuilder. Ensure paths are correct.")
    # This is a critical module, so a simple mock might not be enough for it to be useful.
    # For development, ensure your PYTHONPATH or execution context allows these imports.
    raise # Re-raise to stop if imports fail here

logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class FeatureBuilder:
    def __init__(self, config_obj, logger_obj, market_structure_analyzer: MarketStructureAnalyzer):
        self.config = config_obj
        self.logger = logger_obj
        self.market_analyzer = market_structure_analyzer
        self.indicators_calculator = Indicators()
        self.ml_feature_columns = []

        self.logger.info("FeatureBuilder initialized.")

    def _add_base_ta_indicators(self, df: pd.DataFrame, timeframe_str: str) -> pd.DataFrame:
        """Adds common TA indicators to the given DataFrame (M5 or H1)."""
        self.logger.info(f"Adding base TA indicators for timeframe: {timeframe_str}...")
        df_out = df.copy()

        # ATR
        atr_period = getattr(self.config, 'ATR_PERIOD_INDICATORS', 14) # General ATR
        df_out[f'atr{timeframe_str}'] = self.indicators_calculator.atr(
            df_out[self.config.HIGH_COL], df_out[self.config.LOW_COL], df_out[self.config.CLOSE_COL], period=atr_period
        )

        # EMAs for ML features
        if timeframe_str == self.config.TIMEFRAME_M5_STR:
            ema_short_p = self.config.M5_EMA_SHORT_PERIOD_ML
            ema_long_p = self.config.M5_EMA_LONG_PERIOD_ML
            # Specific EMAs for P1's EMA slope/distance features (e.g., 5 and 20)
            # These might be different from M5_EMA_SHORT/LONG_PERIOD_ML.
            # Let's assume M5_EMA_SHORT/LONG_PERIOD_ML are the ones for these features for now.
            # If P1 used, say, EMA(5) and EMA(20) specifically for slope/distance, add them here.
            # Example: df_out[f'ema5{timeframe_str}'] = self.indicators_calculator.ema(df_out[self.config.CLOSE_COL], period=5)
            #          df_out[f'ema20{timeframe_str}'] = self.indicators_calculator.ema(df_out[self.config.CLOSE_COL], period=20)
        elif timeframe_str == self.config.TIMEFRAME_H1_STR:
            ema_short_p = self.config.H1_EMA_SHORT_PERIOD_ML
            ema_long_p = self.config.H1_EMA_LONG_PERIOD_ML
        else: # Should not happen
            self.logger.error(f"Unknown timeframe_str '{timeframe_str}' for EMA periods."); return df_out
        
        df_out[f'ema_short{timeframe_str}'] = self.indicators_calculator.ema(df_out[self.config.CLOSE_COL], period=ema_short_p)
        df_out[f'ema_long{timeframe_str}'] = self.indicators_calculator.ema(df_out[self.config.CLOSE_COL], period=ema_long_p)

        # RSI
        rsi_period = getattr(self.config, 'RSI_PERIOD', atr_period) # Use ATR_PERIOD_INDICATORS if RSI_PERIOD not set
        df_out[f'rsi{timeframe_str}'] = self.indicators_calculator.rsi(df_out[self.config.CLOSE_COL], period=rsi_period)

        # Volume Features (typically for M5)
        if timeframe_str == self.config.TIMEFRAME_M5_STR:
            vol_roll_period = self.config.M5_VOLUME_ROLLING_AVG_PERIOD
            df_out[f'volume_rolling_avg{timeframe_str}'] = df_out[self.config.VOLUME_COL].rolling(window=vol_roll_period, min_periods=1).mean()
            df_out[f'volume_slope1{timeframe_str}'] = df_out[self.config.VOLUME_COL].diff(1)
        
        self.logger.info(f"Base TA indicators added for {timeframe_str}.")
        return df_out

    def _align_and_merge_h1_context(self, df_m5_with_indicators: pd.DataFrame, df_h1_full_context: pd.DataFrame) -> pd.DataFrame:
        """Aligns H1 data (dynamic SNR and H1 indicators) to M5 data."""
        self.logger.info("Aligning M5 data with H1 context...")
        df_m5_copy = df_m5_with_indicators.copy()

        if df_h1_full_context is None or df_h1_full_context.empty:
            self.logger.warning("H1 full context data is empty. Merged data will lack H1 features.")
            # Define placeholder columns for all expected H1 features on the M5 dataframe
            h1_indicator_cols = [f'ema_short{self.config.TIMEFRAME_H1_STR}', f'ema_long{self.config.TIMEFRAME_H1_STR}',
                                 f'rsi{self.config.TIMEFRAME_H1_STR}', f'atr{self.config.TIMEFRAME_H1_STR}',
                                 self.config.CLOSE_COL + self.config.TIMEFRAME_H1_STR] # For h1_close_col
            h1_snr_cols = self.config.H1_DYNAMIC_SUPPORT_COLS + self.config.H1_DYNAMIC_RESISTANCE_COLS
            placeholder_h1_cols = h1_indicator_cols + h1_snr_cols
            for col in placeholder_h1_cols:
                if col not in df_m5_copy.columns: df_m5_copy[col] = np.nan
            return df_m5_copy

        df_m5_copy['h1_timestamp_ref'] = df_m5_copy.index.floor('H')
        
        # Select H1 columns to merge (all except raw OHLCV if they are not suffixed)
        # H1 dynamic SNR cols are already named like 'h1_dyn_support_1'
        # H1 indicator cols are named like 'ema_shortH1'
        # We also need H1 close for some features, let's ensure it's suffixed if taken from H1.
        df_h1_to_merge = df_h1_full_context.add_suffix(self.config.TIMEFRAME_H1_STR, # Suffix H1's OHLCV
                                        cols_to_suffix=[self.config.OPEN_COL, self.config.HIGH_COL, self.config.LOW_COL, self.config.CLOSE_COL, self.config.VOLUME_COL])

        df_merged = pd.merge(df_m5_copy, df_h1_to_merge,
                             left_on='h1_timestamp_ref', right_index=True,
                             how='left', suffixes=('', '_h1_dup')) # _h1_dup for unexpected clashes

        # Forward fill merged H1 data
        cols_from_h1 = list(df_h1_to_merge.columns)
        for col in cols_from_h1:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill()
        
        df_merged.drop(columns=['h1_timestamp_ref'], inplace=True, errors='ignore')
        self.logger.info(f"Timeframe alignment complete. Merged data shape: {df_merged.shape}")
        return df_merged

    def _create_final_ml_features(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """Creates the final feature vectors for the ML model from the merged data (Project 1 logic)."""
        self.logger.info(f"Creating final ML feature vectors from merged data ({len(df_merged)} rows)...")
        if df_merged.empty: self.logger.warning("Merged DataFrame is empty for final feature creation."); return pd.DataFrame()

        features_df = df_merged.copy()
        self.ml_feature_columns = [] # Reset for this build

        # --- M5 Base Data & Indicators ---
        m5_close = features_df[self.config.CLOSE_COL] # M5 close (unsuffixed)
        m5_open = features_df[self.config.OPEN_COL]
        m5_high = features_df[self.config.HIGH_COL]
        m5_low = features_df[self.config.LOW_COL]
        m5_volume = features_df[self.config.VOLUME_COL]
        
        atr_m5_col = f'atr{self.config.TIMEFRAME_M5_STR}'
        if atr_m5_col not in features_df.columns:
            self.logger.error(f"Critical M5 ATR column '{atr_m5_col}' missing! Adding placeholder."); features_df[atr_m5_col] = 0.00001
        m5_atr = features_df[atr_m5_col].replace(0, np.nan) # For normalization, avoid div by zero

        # Add existing M5 indicators to ML features
        m5_base_indi_cols = [atr_m5_col, f'ema_short{self.config.TIMEFRAME_M5_STR}',
                             f'ema_long{self.config.TIMEFRAME_M5_STR}', f'rsi{self.config.TIMEFRAME_M5_STR}']
        for col in m5_base_indi_cols:
            if col in features_df.columns: self.ml_feature_columns.append(col)

        # --- H1 Dynamic SNR Distance Features ---
        min_dist_sup = pd.Series(np.inf, index=features_df.index)
        for col_name in self.config.H1_DYNAMIC_SUPPORT_COLS: # e.g., ['h1_dyn_support_1', 'h1_dyn_support_2']
            if col_name in features_df.columns:
                s_level = features_df[col_name]
                dist = (m5_close - s_level) / m5_atr
                dist[s_level.isna() | (s_level >= m5_close) | m5_atr.isna()] = np.inf
                min_dist_sup = np.minimum(min_dist_sup, dist)
        features_df[self.config.DIST_TO_H1_DYN_SUPPORT_ATR] = min_dist_sup.replace(np.inf, 999) # 999 for "very far"
        self.ml_feature_columns.append(self.config.DIST_TO_H1_DYN_SUPPORT_ATR)

        min_dist_res = pd.Series(np.inf, index=features_df.index)
        for col_name in self.config.H1_DYNAMIC_RESISTANCE_COLS:
            if col_name in features_df.columns:
                r_level = features_df[col_name]
                dist = (r_level - m5_close) / m5_atr
                dist[r_level.isna() | (r_level <= m5_close) | m5_atr.isna()] = np.inf
                min_dist_res = np.minimum(min_dist_res, dist)
        features_df[self.config.DIST_TO_H1_DYN_RESISTANCE_ATR] = min_dist_res.replace(np.inf, 999)
        self.ml_feature_columns.append(self.config.DIST_TO_H1_DYN_RESISTANCE_ATR)

        # --- M5 EMA Features (e.g., 9/21 or 5/20 based on M5_EMA_SHORT/LONG_PERIOD_ML) ---
        ema_short_m5_col = f'ema_short{self.config.TIMEFRAME_M5_STR}'
        ema_long_m5_col = f'ema_long{self.config.TIMEFRAME_M5_STR}'

        if ema_short_m5_col in features_df.columns:
            features_df[f'dist_close_to_ema_short_m5_atr'] = (m5_close - features_df[ema_short_m5_col]) / m5_atr
            features_df[f'{ema_short_m5_col}_slope1'] = features_df[ema_short_m5_col].diff(1)
            features_df[self.config.M5_EMA_SHORT_SLOPE_NORM_COL] = features_df[f'{ema_short_m5_col}_slope1'] / m5_atr
            self.ml_feature_columns.extend([f'dist_close_to_ema_short_m5_atr', self.config.M5_EMA_SHORT_SLOPE_NORM_COL])
        
        if ema_long_m5_col in features_df.columns:
            features_df[f'dist_close_to_ema_long_m5_atr'] = (m5_close - features_df[ema_long_m5_col]) / m5_atr
            self.ml_feature_columns.append(f'dist_close_to_ema_long_m5_atr')

        if ema_short_m5_col in features_df.columns and ema_long_m5_col in features_df.columns:
            features_df['m5_ema_spread_norm_atr'] = (features_df[ema_short_m5_col] - features_df[ema_long_m5_col]) / m5_atr
            self.ml_feature_columns.append('m5_ema_spread_norm_atr')

        # --- H1 Context Features (from merged H1 indicators) ---
        h1_ema_short_col_merged = f'ema_short{self.config.TIMEFRAME_H1_STR}'
        h1_ema_long_col_merged = f'ema_long{self.config.TIMEFRAME_H1_STR}'
        h1_atr_col_merged = f'atr{self.config.TIMEFRAME_H1_STR}'
        h1_close_col_merged = self.config.CLOSE_COL + self.config.TIMEFRAME_H1_STR # e.g. closeH1

        if all(c in features_df.columns for c in [h1_ema_short_col_merged, h1_ema_long_col_merged, h1_atr_col_merged, h1_close_col_merged]):
            h1_atr_series = features_df[h1_atr_col_merged].replace(0, np.nan)
            features_df['h1_ema_short_above_long_flag'] = (features_df[h1_ema_short_col_merged] > features_df[h1_ema_long_col_merged]).astype(int)
            features_df['h1_ema_spread_norm_atr'] = (features_df[h1_ema_short_col_merged] - features_df[h1_ema_long_col_merged]) / h1_atr_series
            features_df['h1_close_dist_ema_short_atr'] = (features_df[h1_close_col_merged] - features_df[h1_ema_short_col_merged]) / h1_atr_series
            self.ml_feature_columns.extend(['h1_ema_short_above_long_flag', 'h1_ema_spread_norm_atr', 'h1_close_dist_ema_short_atr'])

        # --- M5 Candle Structure Features ---
        features_df['m5_candle_body_norm_atr'] = (m5_close - m5_open).abs() / m5_atr
        features_df['m5_candle_bullish_flag'] = np.sign(m5_close - m5_open).astype(int)
        features_df['m5_upper_wick_norm_atr'] = (m5_high - np.maximum(m5_open, m5_close)) / m5_atr
        features_df['m5_lower_wick_norm_atr'] = (np.minimum(m5_open, m5_close) - m5_low) / m5_atr
        self.ml_feature_columns.extend(['m5_candle_body_norm_atr', 'm5_candle_bullish_flag', 'm5_upper_wick_norm_atr', 'm5_lower_wick_norm_atr'])

        # --- M5 Volume Profile Features ---
        m5_vol_avg_col = f'volume_rolling_avg{self.config.TIMEFRAME_M5_STR}'
        m5_vol_slope_col = f'volume_slope1{self.config.TIMEFRAME_M5_STR}'
        if m5_vol_avg_col in features_df.columns:
            features_df['m5_volume_vs_avg'] = m5_volume / features_df[m5_vol_avg_col].replace(0, np.nan)
            self.ml_feature_columns.append('m5_volume_vs_avg')
        if m5_vol_slope_col in features_df.columns: # This was calculated in _add_base_ta_indicators
            features_df['m5_volume_slope1'] = features_df[m5_vol_slope_col]
            self.ml_feature_columns.append('m5_volume_slope1')

        # --- Session & Time Features ---
        if self.config.APPLY_SESSION_FILTER: # Add as features regardless for model to learn
            features_df[config.IS_LONDON_SESSION_COL] = features_df.index.to_series().apply(
                lambda ts: 1 if (dtime.fromisoformat(self.config.LONDON_SESSION_START) <= ts.time() < dtime.fromisoformat(self.config.LONDON_SESSION_END)) else 0
            )
            features_df[config.IS_NY_SESSION_COL] = features_df.index.to_series().apply(
                lambda ts: 1 if (dtime.fromisoformat(self.config.NY_SESSION_START) <= ts.time() < dtime.fromisoformat(self.config.NY_SESSION_END)) else 0
            )
            self.ml_feature_columns.extend([config.IS_LONDON_SESSION_COL, config.IS_NY_SESSION_COL])

        seconds_in_day = 24 * 60 * 60
        time_seconds = features_df.index.hour * 3600 + features_df.index.minute * 60 + features_df.index.second
        features_df['time_of_day_sin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day)
        features_df['time_of_day_cos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day)
        features_df['day_of_week'] = features_df.index.dayofweek
        self.ml_feature_columns.extend(['time_of_day_sin', 'time_of_day_cos', 'day_of_week'])
        
        # Fill NaNs for all ML feature columns (pipeline imputer will handle this more robustly based on train data)
        # However, some features like slopes might have initial NaNs that should be 0.
        cols_to_fillna_zero = [col for col in self.ml_feature_columns if 'slope' in col or '_flag' in col] # Example
        for col in cols_to_fillna_zero:
            if col in features_df.columns: features_df[col].fillna(0, inplace=True)
        # Other NaNs will be handled by the imputer in the ML pipeline.

        self.logger.info(f"Final ML feature vector creation complete. Shape: {features_df.shape}. Num ML features: {len(self.ml_feature_columns)}")
        return features_df

    def _define_ml_target(self, features_df_with_ohlc: pd.DataFrame, df_m5_ohlcv_full_for_labeling: pd.DataFrame) -> pd.DataFrame:
        """Labels signals based on future price movement (Project 1 logic)."""
        self.logger.info(f"Defining ML target '{self.config.TARGET_COLUMN_NAME}'. Features DF shape: {features_df_with_ohlc.shape}")
        target_col_name = self.config.TARGET_COLUMN_NAME
        if features_df_with_ohlc.empty:
            self.logger.warning("Features DF empty for labeling."); return features_df_with_ohlc.assign(**{target_col_name: 2})
        if df_m5_ohlcv_full_for_labeling.empty:
             self.logger.warning("M5 OHLCV full for labeling is empty."); return features_df_with_ohlc.assign(**{target_col_name: 2})


        atr_m5_col = self.config.M5_ATR_COL_BASE # e.g. 'atrM5'
        if atr_m5_col not in features_df_with_ohlc.columns:
            self.logger.error(f"ATR column '{atr_m5_col}' not found for labeling. Assigning neutral target.")
            return features_df_with_ohlc.assign(**{target_col_name: 2})

        target_labels = np.full(len(features_df_with_ohlc), 2, dtype=int) # Default to neutral
        lookahead = self.config.TARGET_VARIABLE_LOOKAHEAD_CANDLES_M5
        sl_atr_mult = self.config.TARGET_SL_ATR_MULTIPLIER
        tp_rrr = self.config.TARGET_TP_RRR

        df_m5_ohlcv_full_for_labeling = df_m5_ohlcv_full_for_labeling.sort_index()
        if not df_m5_ohlcv_full_for_labeling.index.is_unique:
            df_m5_ohlcv_full_for_labeling = df_m5_ohlcv_full_for_labeling[~df_m5_ohlcv_full_for_labeling.index.duplicated(keep='first')]

        # Iterate using index for potentially faster lookups if needed, or direct iteration
        for i, (timestamp, feature_row) in enumerate(features_df_with_ohlc.iterrows()):
            entry_price = feature_row[self.config.CLOSE_COL]
            atr_val = feature_row[atr_m5_col]

            if pd.isna(entry_price) or pd.isna(atr_val) or atr_val <= 0:
                continue # Keep label as 2 (neutral)

            sl_dist = sl_atr_mult * atr_val
            tp_dist = tp_rrr * sl_dist
            sl_long_price = entry_price - sl_dist
            tp_long_price = entry_price + tp_dist
            
            try: # Find index in the *full* OHLCV data used for looking ahead
                current_idx_in_full_data = df_m5_ohlcv_full_for_labeling.index.get_loc(timestamp)
            except KeyError:
                self.logger.debug(f"Timestamp {timestamp} not found in full M5 OHLCV for labeling. Skipping.")
                continue

            # Define the window for looking ahead in the full OHLCV data
            future_candles_start_idx = current_idx_in_full_data + 1
            future_candles_end_idx = current_idx_in_full_data + 1 + lookahead
            
            if future_candles_end_idx > len(df_m5_ohlcv_full_for_labeling): # Not enough future data
                continue

            future_window = df_m5_ohlcv_full_for_labeling.iloc[future_candles_start_idx:future_candles_end_idx]
            if len(future_window) < lookahead: continue # Should be redundant if above check is correct

            long_win = False; long_loss = False
            for _, future_candle in future_window.iterrows():
                if future_candle[self.config.LOW_COL] <= sl_long_price: long_loss = True; break
                if future_candle[self.config.HIGH_COL] >= tp_long_price: long_win = True; break
            
            if long_win and not long_loss: target_labels[i] = 1
            elif long_loss: target_labels[i] = 0
            # Else, remains 2 (neutral)

        features_df_out = features_df_with_ohlc.copy()
        features_df_out[target_col_name] = target_labels
        self.logger.info(f"ML Target labeling complete. Target dist:\n{features_df_out[target_col_name].value_counts(normalize=True, dropna=False)}")
        return features_df_out

    def _add_backtester_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds features specifically needed by the backtester (M5 pivots, M5 exit EMA)."""
        df_out = df.copy()
        self.logger.info("Adding M5 Pivots and Exit EMA for backtester...")

        m5_pivots_df = self.market_analyzer.get_m5_pivot_levels(df_out[[self.config.OPEN_COL, self.config.HIGH_COL, self.config.LOW_COL, self.config.CLOSE_COL]].copy()) # Pass only OHLC
        if not m5_pivots_df.empty:
            df_out[self.config.M5_PIVOT_SUPPORT_COL] = m5_pivots_df.get(self.config.M5_PIVOT_SUPPORT_COL)
            df_out[self.config.M5_PIVOT_RESISTANCE_COL] = m5_pivots_df.get(self.config.M5_PIVOT_RESISTANCE_COL)
        else:
            df_out[self.config.M5_PIVOT_SUPPORT_COL] = np.nan
            df_out[self.config.M5_PIVOT_RESISTANCE_COL] = np.nan

        df_out[self.config.M5_EXIT_EMA_COL] = self.indicators_calculator.ema(
            df_out[self.config.CLOSE_COL], period=self.config.M5_EXIT_EMA_PERIOD
        )
        return df_out

    # --- Main Orchestrator Methods ---
    def build_features_and_labels(self, df_m5_raw: pd.DataFrame, df_h1_raw: pd.DataFrame = None) -> pd.DataFrame:
        self.logger.info(f"Starting full feature and label building (M5: {len(df_m5_raw)} rows)...")
        if df_m5_raw is None or df_m5_raw.empty: self.logger.error("M5 raw data empty."); return pd.DataFrame()

        df_m5_indic = self._add_base_ta_indicators(df_m5_raw, self.config.TIMEFRAME_M5_STR)
        df_h1_full_context = None
        if df_h1_raw is not None and not df_h1_raw.empty:
            df_h1_indic = self._add_base_ta_indicators(df_h1_raw, self.config.TIMEFRAME_H1_STR)
            dynamic_h1_snr_df = self.market_analyzer.calculate_all_dynamic_h1_snr(df_h1_indic.copy()) # Pass copy
            df_h1_full_context = pd.merge(df_h1_indic, dynamic_h1_snr_df, left_index=True, right_index=True, how='left')
        
        df_merged = self._align_and_merge_h1_context(df_m5_indic, df_h1_full_context)
        df_features_ml_base = self._create_final_ml_features(df_merged)
        df_with_backtest_cols = self._add_backtester_specific_features(df_features_ml_base)
        df_labeled_final = self._define_ml_target(df_with_backtest_cols, df_m5_raw) # Use original m5_raw for labeling lookahead
        
        self.ml_feature_columns = sorted(list(set(col for col in self.ml_feature_columns if col in df_labeled_final.columns))) # Ensure they exist
        self.logger.info(f"Full feature/label building complete. Final DF shape: {df_labeled_final.shape}. ML Features: {len(self.ml_feature_columns)}")
        return df_labeled_final

    def build_features_for_live(self, df_m5_live_segment: pd.DataFrame, df_h1_live_segment: pd.DataFrame = None) -> pd.DataFrame:
        self.logger.info(f"Building features for live data (M5: {len(df_m5_live_segment)} rows)...")
        if df_m5_live_segment is None or df_m5_live_segment.empty: self.logger.error("Live M5 segment empty."); return pd.DataFrame()

        df_m5_indic = self._add_base_ta_indicators(df_m5_live_segment, self.config.TIMEFRAME_M5_STR)
        df_h1_full_context = None
        if df_h1_live_segment is not None and not df_h1_live_segment.empty:
            df_h1_indic = self._add_base_ta_indicators(df_h1_live_segment, self.config.TIMEFRAME_H1_STR)
            # For live, calculate SNR for the latest H1 state based on its lookback window
            # calculate_all_dynamic_h1_snr will process the segment and the last row will have the latest SNR
            dynamic_h1_snr_df = self.market_analyzer.calculate_all_dynamic_h1_snr(df_h1_indic.copy())
            df_h1_full_context = pd.merge(df_h1_indic, dynamic_h1_snr_df, left_index=True, right_index=True, how='left')
        
        df_merged = self._align_and_merge_h1_context(df_m5_indic, df_h1_full_context)
        df_features_ml_live = self._create_final_ml_features(df_merged)
        # For live, we might also need backtester specific features if signal rules or SL/TP use them directly
        df_features_live_final = self._add_backtester_specific_features(df_features_ml_live)

        self.ml_feature_columns = sorted(list(set(col for col in self.ml_feature_columns if col in df_features_live_final.columns)))
        self.logger.info(f"Live feature building complete. Output DF shape: {df_features_live_final.shape}. ML Features: {len(self.ml_feature_columns)}")
        return df_features_live_final # Return all columns; SignalProcessor will use what it needs

    def get_ml_feature_columns(self) -> list:
        if not self.ml_feature_columns:
            self.logger.warning("ML feature columns list is empty. Call build_features first.")
        return self.ml_feature_columns