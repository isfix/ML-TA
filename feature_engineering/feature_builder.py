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
    raise

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
        self.logger.debug(f"Adding base TA indicators for timeframe: {timeframe_str}...") # Changed to debug
        df_out = df.copy()

        atr_period = getattr(self.config, 'ATR_PERIOD_INDICATORS', 14)
        df_out[f'atr{timeframe_str}'] = self.indicators_calculator.atr(
            df_out[self.config.HIGH_COL], df_out[self.config.LOW_COL], df_out[self.config.CLOSE_COL], period=atr_period
        )

        if timeframe_str == self.config.TIMEFRAME_M5_STR:
            ema_short_p = self.config.M5_EMA_SHORT_PERIOD_ML
            ema_long_p = self.config.M5_EMA_LONG_PERIOD_ML
        elif timeframe_str == self.config.TIMEFRAME_H1_STR:
            ema_short_p = self.config.H1_EMA_SHORT_PERIOD_ML
            ema_long_p = self.config.H1_EMA_LONG_PERIOD_ML
        else:
            self.logger.error(f"Unknown timeframe_str '{timeframe_str}' for EMA periods."); return df_out
        
        df_out[f'ema_short{timeframe_str}'] = self.indicators_calculator.ema(df_out[self.config.CLOSE_COL], period=ema_short_p)
        df_out[f'ema_long{timeframe_str}'] = self.indicators_calculator.ema(df_out[self.config.CLOSE_COL], period=ema_long_p)

        rsi_period = getattr(self.config, 'RSI_PERIOD', atr_period)
        df_out[f'rsi{timeframe_str}'] = self.indicators_calculator.rsi(df_out[self.config.CLOSE_COL], period=rsi_period)

        if timeframe_str == self.config.TIMEFRAME_M5_STR:
            vol_roll_period = self.config.M5_VOLUME_ROLLING_AVG_PERIOD
            df_out[f'volume_rolling_avg{timeframe_str}'] = df_out[self.config.VOLUME_COL].rolling(window=vol_roll_period, min_periods=1).mean()
            df_out[f'volume_slope1{timeframe_str}'] = df_out[self.config.VOLUME_COL].diff(1).fillna(0)

        self.logger.debug(f"Base TA indicators added for {timeframe_str}.") # Changed to debug
        return df_out

    def _align_and_merge_h1_context(self, df_m5_with_indicators: pd.DataFrame, df_h1_full_context: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Aligning M5 data with H1 context...") # Changed to debug
        df_m5_copy = df_m5_with_indicators.copy()

        if df_h1_full_context is None or df_h1_full_context.empty:
            self.logger.warning("H1 full context data is empty. Merged data will lack H1 features.")
            h1_indicator_cols = [f'ema_short{self.config.TIMEFRAME_H1_STR}', f'ema_long{self.config.TIMEFRAME_H1_STR}',
                                 f'rsi{self.config.TIMEFRAME_H1_STR}', f'atr{self.config.TIMEFRAME_H1_STR}',
                                 self.config.CLOSE_COL + self.config.TIMEFRAME_H1_STR]
            h1_snr_cols = self.config.H1_DYNAMIC_SUPPORT_COLS + self.config.H1_DYNAMIC_RESISTANCE_COLS
            placeholder_h1_cols = h1_indicator_cols + h1_snr_cols
            for col in placeholder_h1_cols:
                if col not in df_m5_copy.columns: df_m5_copy[col] = np.nan
            return df_m5_copy

        df_m5_copy['h1_timestamp_ref'] = df_m5_copy.index.floor('H')
        
        df_h1_to_merge = df_h1_full_context.copy()
        ohlcv_to_suffix_h1 = {
            self.config.OPEN_COL: self.config.OPEN_COL + self.config.TIMEFRAME_H1_STR,
            self.config.HIGH_COL: self.config.HIGH_COL + self.config.TIMEFRAME_H1_STR,
            self.config.LOW_COL: self.config.LOW_COL + self.config.TIMEFRAME_H1_STR,
            self.config.CLOSE_COL: self.config.CLOSE_COL + self.config.TIMEFRAME_H1_STR,
            self.config.VOLUME_COL: self.config.VOLUME_COL + self.config.TIMEFRAME_H1_STR,
        }
        # Only rename if columns exist and are not already suffixed (less likely for OHLCV)
        cols_to_rename_h1 = {k: v for k, v in ohlcv_to_suffix_h1.items() if k in df_h1_to_merge.columns}
        df_h1_to_merge.rename(columns=cols_to_rename_h1, inplace=True)

        df_merged = pd.merge(df_m5_copy, df_h1_to_merge,
                             left_on='h1_timestamp_ref', right_index=True,
                             how='left', suffixes=('', '_h1_dup'))

        cols_from_h1 = list(df_h1_to_merge.columns)
        for col in cols_from_h1:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].ffill()
        
        df_merged.drop(columns=['h1_timestamp_ref'], inplace=True, errors='ignore')
        self.logger.debug(f"Timeframe alignment complete. Merged data shape: {df_merged.shape}") # Changed to debug
        return df_merged

    def _create_final_ml_features(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Creating final ML feature vectors from merged data ({len(df_merged)} rows)...") # Changed to debug
        if df_merged.empty: self.logger.warning("Merged DataFrame is empty for final feature creation."); return pd.DataFrame()

        features_df = df_merged.copy()
        # Reset ml_feature_columns at the start of this specific feature creation process
        # If build_all_features... calls this, it should manage its own list.
        # For clarity, this method will return the df and the list of features it thinks are ML features.
        current_ml_features = [] 

        m5_close = features_df[self.config.CLOSE_COL]
        m5_open = features_df[self.config.OPEN_COL]
        m5_high = features_df[self.config.HIGH_COL]
        m5_low = features_df[self.config.LOW_COL]
        m5_volume = features_df[self.config.VOLUME_COL]
        
        atr_m5_col = f'atr{self.config.TIMEFRAME_M5_STR}'
        if atr_m5_col not in features_df.columns:
            self.logger.error(f"Critical M5 ATR column '{atr_m5_col}' missing! Adding placeholder."); features_df[atr_m5_col] = 0.00001
        m5_atr = features_df[atr_m5_col].replace(0, np.nan)

        m5_base_indi_cols = [atr_m5_col, f'ema_short{self.config.TIMEFRAME_M5_STR}',
                             f'ema_long{self.config.TIMEFRAME_M5_STR}', f'rsi{self.config.TIMEFRAME_M5_STR}']
        for col in m5_base_indi_cols:
            if col in features_df.columns: current_ml_features.append(col)

        min_dist_sup = pd.Series(np.inf, index=features_df.index)
        for col_name in self.config.H1_DYNAMIC_SUPPORT_COLS:
            if col_name in features_df.columns:
                s_level = features_df[col_name]
                dist = (m5_close - s_level) / m5_atr
                dist[s_level.isna() | (s_level >= m5_close) | m5_atr.isna()] = np.inf
                min_dist_sup = np.minimum(min_dist_sup, dist)
        features_df[self.config.DIST_TO_H1_DYN_SUPPORT_ATR] = min_dist_sup.replace(np.inf, 999)
        current_ml_features.append(self.config.DIST_TO_H1_DYN_SUPPORT_ATR)

        min_dist_res = pd.Series(np.inf, index=features_df.index)
        for col_name in self.config.H1_DYNAMIC_RESISTANCE_COLS:
            if col_name in features_df.columns:
                r_level = features_df[col_name]
                dist = (r_level - m5_close) / m5_atr
                dist[r_level.isna() | (r_level <= m5_close) | m5_atr.isna()] = np.inf
                min_dist_res = np.minimum(min_dist_res, dist)
        features_df[self.config.DIST_TO_H1_DYN_RESISTANCE_ATR] = min_dist_res.replace(np.inf, 999)
        current_ml_features.append(self.config.DIST_TO_H1_DYN_RESISTANCE_ATR)

        ema_short_m5_col = f'ema_short{self.config.TIMEFRAME_M5_STR}'
        ema_long_m5_col = f'ema_long{self.config.TIMEFRAME_M5_STR}'

        if ema_short_m5_col in features_df.columns:
            features_df[f'dist_close_to_ema_short_m5_atr'] = (m5_close - features_df[ema_short_m5_col]) / m5_atr
            features_df[f'{ema_short_m5_col}_slope1'] = features_df[ema_short_m5_col].diff(1).fillna(0)
            features_df[self.config.M5_EMA_SHORT_SLOPE_NORM_COL] = features_df[f'{ema_short_m5_col}_slope1'] / m5_atr
            current_ml_features.extend([f'dist_close_to_ema_short_m5_atr', self.config.M5_EMA_SHORT_SLOPE_NORM_COL])
        
        if ema_long_m5_col in features_df.columns:
            features_df[f'dist_close_to_ema_long_m5_atr'] = (m5_close - features_df[ema_long_m5_col]) / m5_atr
            current_ml_features.append(f'dist_close_to_ema_long_m5_atr')

        if ema_short_m5_col in features_df.columns and ema_long_m5_col in features_df.columns:
            features_df['m5_ema_spread_norm_atr'] = (features_df[ema_short_m5_col] - features_df[ema_long_m5_col]) / m5_atr
            current_ml_features.append('m5_ema_spread_norm_atr')

        h1_ema_short_col_merged = f'ema_short{self.config.TIMEFRAME_H1_STR}'
        h1_ema_long_col_merged = f'ema_long{self.config.TIMEFRAME_H1_STR}'
        h1_atr_col_merged = f'atr{self.config.TIMEFRAME_H1_STR}'
        h1_close_col_merged = self.config.CLOSE_COL + self.config.TIMEFRAME_H1_STR

        if all(c in features_df.columns for c in [h1_ema_short_col_merged, h1_ema_long_col_merged, h1_atr_col_merged, h1_close_col_merged]):
            h1_atr_series = features_df[h1_atr_col_merged].replace(0, np.nan)
            features_df['h1_ema_short_above_long_flag'] = (features_df[h1_ema_short_col_merged] > features_df[h1_ema_long_col_merged]).astype(int)
            features_df['h1_ema_spread_norm_atr'] = (features_df[h1_ema_short_col_merged] - features_df[h1_ema_long_col_merged]) / h1_atr_series
            features_df['h1_close_dist_ema_short_atr'] = (features_df[h1_close_col_merged] - features_df[h1_ema_short_col_merged]) / h1_atr_series
            current_ml_features.extend(['h1_ema_short_above_long_flag', 'h1_ema_spread_norm_atr', 'h1_close_dist_ema_short_atr'])

        features_df['m5_candle_body_norm_atr'] = (m5_close - m5_open).abs() / m5_atr
        features_df['m5_candle_bullish_flag'] = np.sign(m5_close - m5_open).astype(int)
        features_df['m5_upper_wick_norm_atr'] = (m5_high - np.maximum(m5_open, m5_close)) / m5_atr
        features_df['m5_lower_wick_norm_atr'] = (np.minimum(m5_open, m5_close) - m5_low) / m5_atr
        current_ml_features.extend(['m5_candle_body_norm_atr', 'm5_candle_bullish_flag', 'm5_upper_wick_norm_atr', 'm5_lower_wick_norm_atr'])

        m5_vol_avg_col = f'volume_rolling_avg{self.config.TIMEFRAME_M5_STR}'
        m5_vol_slope_col = f'volume_slope1{self.config.TIMEFRAME_M5_STR}'
        if m5_vol_avg_col in features_df.columns and self.config.VOLUME_COL in features_df.columns: # M5 volume is unsuffixed
            features_df['m5_volume_vs_avg'] = features_df[self.config.VOLUME_COL] / features_df[m5_vol_avg_col].replace(0, np.nan)
            current_ml_features.append('m5_volume_vs_avg')
        if m5_vol_slope_col in features_df.columns:
            features_df['m5_volume_slope1'] = features_df[m5_vol_slope_col]
            current_ml_features.append('m5_volume_slope1')

        if getattr(self.config, 'APPLY_SESSION_FILTER', True):
            if not isinstance(features_df.index, pd.DatetimeIndex):
                self.logger.error("Index is not DatetimeIndex for session features.")
            else:
                # Ensure IS_LONDON_SESSION_COL and IS_NY_SESSION_COL are defined in config
                london_col = getattr(self.config, 'IS_LONDON_SESSION_COL', 'is_london_session')
                ny_col = getattr(self.config, 'IS_NY_SESSION_COL', 'is_ny_session')
                features_df[london_col] = features_df.index.to_series().apply(
                    lambda ts: 1 if (dtime.fromisoformat(self.config.LONDON_SESSION_START) <= ts.time() < dtime.fromisoformat(self.config.LONDON_SESSION_END)) else 0
                )
                features_df[ny_col] = features_df.index.to_series().apply(
                    lambda ts: 1 if (dtime.fromisoformat(self.config.NY_SESSION_START) <= ts.time() < dtime.fromisoformat(self.config.NY_SESSION_END)) else 0
                )
                current_ml_features.extend([london_col, ny_col])

        if isinstance(features_df.index, pd.DatetimeIndex):
            seconds_in_day = 24 * 60 * 60
            time_seconds = features_df.index.hour * 3600 + features_df.index.minute * 60 + features_df.index.second
            features_df['time_of_day_sin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day)
            features_df['time_of_day_cos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day)
            features_df['day_of_week'] = features_df.index.dayofweek
            current_ml_features.extend(['time_of_day_sin', 'time_of_day_cos', 'day_of_week'])
        
        for col in current_ml_features: # Use current_ml_features list
            if col not in features_df.columns: features_df[col] = 0.0 # Ensure column exists
            else: features_df[col].fillna(0.0, inplace=True) # Fill NaNs for these ML features

        # Store the identified ML features in the instance variable
        self.ml_feature_columns = sorted(list(set(current_ml_features)))

        self.logger.debug(f"Final ML feature vector creation complete. Shape: {features_df.shape}. Num ML features: {len(self.ml_feature_columns)}") # Changed to debug
        return features_df

    def _define_ml_target(self, features_df_with_ohlc: pd.DataFrame, df_m5_ohlcv_full_for_labeling: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Defining ML target '{self.config.TARGET_COLUMN_NAME}'. Features DF shape: {features_df_with_ohlc.shape}")
        target_col_name = self.config.TARGET_COLUMN_NAME
        if features_df_with_ohlc.empty: return features_df_with_ohlc.assign(**{target_col_name: 2})
        if df_m5_ohlcv_full_for_labeling.empty: return features_df_with_ohlc.assign(**{target_col_name: 2})

        atr_m5_col = f'atr{self.config.TIMEFRAME_M5_STR}'
        if atr_m5_col not in features_df_with_ohlc.columns:
            self.logger.error(f"ATR column '{atr_m5_col}' not found for labeling."); return features_df_with_ohlc.assign(**{target_col_name: 2})

        target_labels = np.full(len(features_df_with_ohlc), 2, dtype=int)
        lookahead = self.config.TARGET_VARIABLE_LOOKAHEAD_CANDLES_M5
        sl_atr_mult = self.config.TARGET_SL_ATR_MULTIPLIER
        tp_rrr = self.config.TARGET_TP_RRR

        df_m5_ohlcv_full_for_labeling = df_m5_ohlcv_full_for_labeling.sort_index()
        if not df_m5_ohlcv_full_for_labeling.index.is_unique:
            df_m5_ohlcv_full_for_labeling = df_m5_ohlcv_full_for_labeling[~df_m5_ohlcv_full_for_labeling.index.duplicated(keep='first')]

        for i, timestamp_idx in enumerate(features_df_with_ohlc.index):
            feature_row = features_df_with_ohlc.loc[timestamp_idx]
            entry_price = feature_row[self.config.CLOSE_COL]
            atr_val = feature_row[atr_m5_col]
            outcome = 2 

            if pd.isna(entry_price) or pd.isna(atr_val) or atr_val <= 0: continue

            sl_dist = sl_atr_mult * atr_val; tp_dist = tp_rrr * sl_dist
            sl_long_price = entry_price - sl_dist; tp_long_price = entry_price + tp_dist
            
            try: current_idx_in_full_data = df_m5_ohlcv_full_for_labeling.index.get_loc(timestamp_idx)
            except KeyError: continue

            future_candles_start_idx = current_idx_in_full_data + 1
            future_candles_end_idx = current_idx_in_full_data + 1 + lookahead
            if future_candles_end_idx > len(df_m5_ohlcv_full_for_labeling): continue

            future_window = df_m5_ohlcv_full_for_labeling.iloc[future_candles_start_idx:future_candles_end_idx]
            if len(future_window) < lookahead: continue

            long_win = False; long_loss = False
            for _, future_candle in future_window.iterrows():
                if future_candle[self.config.LOW_COL] <= sl_long_price: long_loss = True; break
                if future_candle[self.config.HIGH_COL] >= tp_long_price: long_win = True; break
            
            if long_win and not long_loss: outcome = 1
            elif long_loss: outcome = 0
            target_labels[i] = outcome

        features_df_out = features_df_with_ohlc.copy()
        features_df_out[target_col_name] = target_labels
        self.logger.info(f"ML Target labeling complete. Target dist:\n{features_df_out[target_col_name].value_counts(normalize=True, dropna=False)}")
        return features_df_out

    def _add_backtester_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        self.logger.debug("Adding M5 Pivots and Exit EMA for backtester...") # Changed to debug

        m5_pivots_df = self.market_analyzer.get_m5_pivot_levels(df_out[[self.config.OPEN_COL, self.config.HIGH_COL, self.config.LOW_COL, self.config.CLOSE_COL]].copy())
        if not m5_pivots_df.empty:
            df_out[self.config.M5_PIVOT_SUPPORT_COL] = m5_pivots_df.get(self.config.M5_PIVOT_SUPPORT_COL)
            df_out[self.config.M5_PIVOT_RESISTANCE_COL] = m5_pivots_df.get(self.config.M5_PIVOT_RESISTANCE_COL)
        else:
            df_out[self.config.M5_PIVOT_SUPPORT_COL] = np.nan
            df_out[self.config.M5_PIVOT_RESISTANCE_COL] = np.nan
        
        # Ensure M5_EXIT_EMA_COL is correctly formed using M5_EXIT_EMA_PERIOD from config
        m5_exit_ema_col_name = getattr(self.config, 'M5_EXIT_EMA_COL', f'm5_exit_ema_{self.config.M5_EXIT_EMA_PERIOD}')
        df_out[m5_exit_ema_col_name] = self.indicators_calculator.ema(
            df_out[self.config.CLOSE_COL], period=self.config.M5_EXIT_EMA_PERIOD
        )
        self.logger.debug("M5 Pivots and Exit EMA added for backtester.") # Changed to debug
        return df_out

    # --- Main Orchestrator Methods ---
    def build_features_and_labels(self, df_m5_raw: pd.DataFrame, df_h1_raw: pd.DataFrame = None) -> pd.DataFrame:
        self.logger.info(f"Starting full feature and label building (M5: {len(df_m5_raw)} rows)...")
        if df_m5_raw is None or df_m5_raw.empty: self.logger.error("M5 raw data empty."); return pd.DataFrame()

        # Ensure raw DFs have datetime index
        if not isinstance(df_m5_raw.index, pd.DatetimeIndex): df_m5_raw.index = pd.to_datetime(df_m5_raw.index, utc=True)
        if df_h1_raw is not None and not df_h1_raw.empty and not isinstance(df_h1_raw.index, pd.DatetimeIndex):
            df_h1_raw.index = pd.to_datetime(df_h1_raw.index, utc=True)


        df_m5_indic = self._add_base_ta_indicators(df_m5_raw, self.config.TIMEFRAME_M5_STR)
        df_h1_full_context = None
        if df_h1_raw is not None and not df_h1_raw.empty:
            df_h1_indic = self._add_base_ta_indicators(df_h1_raw, self.config.TIMEFRAME_H1_STR)
            dynamic_h1_snr_df = self.market_analyzer.calculate_all_dynamic_h1_snr(df_h1_indic.copy())
            df_h1_full_context = pd.merge(df_h1_indic, dynamic_h1_snr_df, left_index=True, right_index=True, how='left')
        
        df_merged = self._align_and_merge_h1_context(df_m5_indic, df_h1_full_context)
        df_features_ml_base = self._create_final_ml_features(df_merged) # This populates self.ml_feature_columns
        df_with_backtest_cols = self._add_backtester_specific_features(df_features_ml_base)
        df_labeled_final = self._define_ml_target(df_with_backtest_cols, df_m5_raw)
        
        # Finalize self.ml_feature_columns based on what's actually in the df
        self.ml_feature_columns = sorted(list(set(col for col in self.ml_feature_columns if col in df_labeled_final.columns)))
        self.logger.info(f"Full feature/label building complete. Final DF shape: {df_labeled_final.shape}. ML Features: {len(self.ml_feature_columns)}")
        return df_labeled_final

    def build_all_features_for_trading_or_backtesting(self, df_m5_raw: pd.DataFrame, df_h1_raw: pd.DataFrame = None) -> pd.DataFrame:
        """Builds all features needed for trading/backtesting, WITHOUT ML target labeling."""
        self.logger.info(f"Building all features for trading/backtesting (M5: {len(df_m5_raw)} rows)...")
        if df_m5_raw is None or df_m5_raw.empty: self.logger.error("M5 raw data empty."); return pd.DataFrame()

        if not isinstance(df_m5_raw.index, pd.DatetimeIndex): df_m5_raw.index = pd.to_datetime(df_m5_raw.index, utc=True)
        if df_h1_raw is not None and not df_h1_raw.empty and not isinstance(df_h1_raw.index, pd.DatetimeIndex):
            df_h1_raw.index = pd.to_datetime(df_h1_raw.index, utc=True)

        df_m5_indic = self._add_base_ta_indicators(df_m5_raw, self.config.TIMEFRAME_M5_STR)
        df_h1_full_context = None
        if df_h1_raw is not None and not df_h1_raw.empty:
            df_h1_indic = self._add_base_ta_indicators(df_h1_raw, self.config.TIMEFRAME_H1_STR)
            dynamic_h1_snr_df = self.market_analyzer.calculate_all_dynamic_h1_snr(df_h1_indic.copy())
            df_h1_full_context = pd.merge(df_h1_indic, dynamic_h1_snr_df, left_index=True, right_index=True, how='left')
        
        df_merged = self._align_and_merge_h1_context(df_m5_indic, df_h1_full_context)
        df_features_ml_base = self._create_final_ml_features(df_merged)
        df_with_backtest_cols = self._add_backtester_specific_features(df_features_ml_base)
        
        self.ml_feature_columns = sorted(list(set(col for col in self.ml_feature_columns if col in df_with_backtest_cols.columns)))
        self.logger.info(f"All features for trading/backtesting built. Final DF shape: {df_with_backtest_cols.shape}. ML Features: {len(self.ml_feature_columns)}")
        return df_with_backtest_cols

    def build_features_for_live_single_candle(self, latest_m5_segment: pd.DataFrame, latest_h1_segment: pd.DataFrame = None) -> pd.Series | None:
        self.logger.debug(f"Building features for latest live candle (M5 seg: {len(latest_m5_segment)} rows)...")
        if latest_m5_segment is None or latest_m5_segment.empty:
            self.logger.error("Live M5 segment empty for single candle feature build."); return None
        
        # Ensure segments have datetime index
        if not isinstance(latest_m5_segment.index, pd.DatetimeIndex): latest_m5_segment.index = pd.to_datetime(latest_m5_segment.index, utc=True)
        if latest_h1_segment is not None and not latest_h1_segment.empty and not isinstance(latest_h1_segment.index, pd.DatetimeIndex):
            latest_h1_segment.index = pd.to_datetime(latest_h1_segment.index, utc=True)

        all_features_df = self.build_all_features_for_trading_or_backtesting(latest_m5_segment, latest_h1_segment)
        
        if all_features_df.empty:
            self.logger.warning("Building all features for live segment returned empty df.")
            return None
            
        all_features_df.sort_index(inplace=True) # Ensure sorted before iloc[-1]
        if all_features_df.empty: # Check again after sort, though unlikely to change emptiness
            self.logger.warning("Features df became empty after sort_index in live single candle build.")
            return None
        return all_features_df.iloc[-1]

    def get_ml_feature_columns(self) -> list:
        if not self.ml_feature_columns:
            self.logger.warning("ML feature columns list is empty. Call a build method first.")
        # Return a copy to prevent external modification
        return list(self.ml_feature_columns)