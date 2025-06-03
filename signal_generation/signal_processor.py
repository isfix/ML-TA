# signal_generation/signal_processor.py
"""
Processes features to generate trading signals.
Combines rule-based candidate identification (dynamic H1 SNR context + M5 EMA slope)
with ML model confirmation.
Based on Project 1's SignalProcessor.
"""
import pandas as pd
import numpy as np

# Assuming config and logging_utils are accessible
try:
    import config
    from utilities import logging_utils
except ImportError:
    print("Warning: Could not perform standard config/utilities import in SignalProcessor. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Mock config for standalone testing
    class MockConfig:
        LOG_FILE_APP = "signal_processor_temp.log"
        # SNR Proximity
        H1_SUPPORT_PROXIMITY_THRESHOLD_ATR = 0.75
        H1_RESISTANCE_PROXIMITY_THRESHOLD_ATR = 0.75
        # M5 EMA Slope
        M5_EMA_SLOPE_THRESHOLD = 0.005
        M5_EMA_SHORT_SLOPE_NORM_COL = 'm5_ema_short_slope_norm' # Example name
        # ML Confirmation
        ML_BUY_CONFIRM_THRESHOLD = 0.60
        ML_SELL_CONFIRM_THRESHOLD = 0.60
        # Feature names for distance to dynamic SNR
        DIST_TO_H1_DYN_SUPPORT_ATR = 'dist_to_h1_dyn_support_atr'
        DIST_TO_H1_DYN_RESISTANCE_ATR = 'dist_to_h1_dyn_resistance_atr'
        # Session Filter
        APPLY_SESSION_FILTER = True
        IS_LONDON_SESSION_COL = 'is_london_session' # Assuming feature_builder creates this
        IS_NY_SESSION_COL = 'is_ny_session'

    if 'config' not in locals() and 'config' not in globals():
        config = MockConfig()
        logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)
else: # Standard import successful
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class SignalProcessor:
    def __init__(self, config_obj, logger_obj, ml_model=None, scaler_pipeline_component=None, feature_names_for_model=None):
        self.config = config_obj
        self.logger = logger_obj
        self.ml_model_pipeline = ml_model # This is now the entire fitted pipeline
        self.scaler = scaler_pipeline_component # This might be None if pipeline handles scaling
        self.feature_names_from_training = feature_names_for_model # Expected feature names for the model

        self.logger.info("SignalProcessor initialized.")
        if self.ml_model_pipeline:
            self.logger.info(f"ML Model Pipeline loaded: {type(self.ml_model_pipeline).__name__}")
        else:
            self.logger.warning("SignalProcessor initialized WITHOUT an ML model pipeline.")
        if self.feature_names_from_training:
            self.logger.info(f"Expected {len(self.feature_names_from_training)} features for model: {self.feature_names_from_training[:5]}...")
        else:
            self.logger.warning("SignalProcessor initialized WITHOUT expected feature names. Predictions may be unreliable.")

    def _apply_session_filter(self, latest_features_series: pd.Series) -> bool:
        """Checks if the current candle is within an active trading session if filtering is enabled."""
        if not getattr(self.config, 'APPLY_SESSION_FILTER', False):
            return True # Filter disabled, always pass

        is_london = latest_features_series.get(getattr(self.config, 'IS_LONDON_SESSION_COL', 'is_london_session'), 0)
        is_ny = latest_features_series.get(getattr(self.config, 'IS_NY_SESSION_COL', 'is_ny_session'), 0)

        if is_london or is_ny:
            return True
        
        self.logger.debug(f"[{latest_features_series.name}] Candle outside of configured London/NY sessions. Signal filtered out.")
        return False

    def check_candidate_entry_conditions(self, latest_features_series: pd.Series) -> str | None:
        """
        Checks for rule-based candidate entry conditions based on Project 1's logic:
        H1 Dynamic SNR proximity + M5 EMA slope.
        Applies session filter.

        Args:
            latest_features_series (pd.Series): Latest computed features for the current M5 candle.
        Returns:
            str: 'BUY', 'SELL', or None.
        """
        timestamp = latest_features_series.name
        self.logger.debug(f"[{timestamp}] Checking candidate entry conditions...")

        if not self._apply_session_filter(latest_features_series):
            return None # Filtered out by session

        # Features from Project 1's logic
        dist_to_h1_support = latest_features_series.get(self.config.DIST_TO_H1_DYN_SUPPORT_ATR, 999)
        dist_to_h1_resistance = latest_features_series.get(self.config.DIST_TO_H1_DYN_RESISTANCE_ATR, 999)
        m5_ema_short_slope_norm = latest_features_series.get(self.config.M5_EMA_SHORT_SLOPE_NORM_COL, 0)

        # Thresholds from config
        h1_sup_prox_atr = self.config.H1_SUPPORT_PROXIMITY_THRESHOLD_ATR
        h1_res_prox_atr = self.config.H1_RESISTANCE_PROXIMITY_THRESHOLD_ATR
        m5_ema_slope_thresh = self.config.M5_EMA_SLOPE_THRESHOLD

        self.logger.debug(f"[{timestamp}] Candidate Check Features: DistToH1Sup={dist_to_h1_support:.3f} (Thresh: <{h1_sup_prox_atr}), "
                          f"DistToH1Res={dist_to_h1_resistance:.3f} (Thresh: <{h1_res_prox_atr}), "
                          f"M5EmaSlopeNorm={m5_ema_short_slope_norm:.4f} (Thresh: >+/-{m5_ema_slope_thresh})")

        # BUY Candidate: Price near H1 Dynamic Support AND M5 EMA short-term slope is bullish.
        is_near_support = dist_to_h1_support < h1_sup_prox_atr
        is_m5_slope_bullish = m5_ema_short_slope_norm > m5_ema_slope_thresh

        if is_near_support and is_m5_slope_bullish:
            self.logger.info(f"[{timestamp}] Candidate BUY signal: Near H1 Dyn Support (DistATR: {dist_to_h1_support:.3f}) AND M5 EMA slope bullish (NormSlope: {m5_ema_short_slope_norm:.4f}).")
            return "BUY"

        # SELL Candidate: Price near H1 Dynamic Resistance AND M5 EMA short-term slope is bearish.
        is_near_resistance = dist_to_h1_resistance < h1_res_prox_atr
        is_m5_slope_bearish = m5_ema_short_slope_norm < -m5_ema_slope_thresh # Note negative threshold

        if is_near_resistance and is_m5_slope_bearish:
            self.logger.info(f"[{timestamp}] Candidate SELL signal: Near H1 Dyn Resistance (DistATR: {dist_to_h1_resistance:.3f}) AND M5 EMA slope bearish (NormSlope: {m5_ema_short_slope_norm:.4f}).")
            return "SELL"

        self.logger.debug(f"[{timestamp}] No strong rule-based candidate signal found.")
        return None

    def generate_ml_confirmed_signal(self, latest_features_series: pd.Series, candidate_direction: str) -> dict | None:
        """
        Confirms a rule-based candidate signal using the ML model pipeline.

        Args:
            latest_features_series (pd.Series): Latest feature vector for the M5 candle.
            candidate_direction (str): 'BUY' or 'SELL' from rule-based check.

        Returns:
            dict or None: {'signal': 'BUY'/'SELL', 'probability': prob, 'timestamp': ts} or None.
        """
        if not self.ml_model_pipeline:
            self.logger.warning("ML model pipeline not available. Cannot provide ML confirmation.")
            # Fallback: Could allow trading based on rules only if configured, but P1 implies ML confirmation.
            return None
        if not self.feature_names_from_training:
            self.logger.error("Feature names from training not available. Cannot align data for ML prediction.")
            return None

        timestamp = latest_features_series.name
        self.logger.debug(f"[{timestamp}] Attempting ML confirmation for candidate '{candidate_direction}' signal.")

        # Prepare features for prediction: DataFrame with one row, columns aligned with training.
        try:
            # Ensure all expected feature names are present in the input series
            missing_features = set(self.feature_names_from_training) - set(latest_features_series.index)
            if missing_features:
                self.logger.error(f"[{timestamp}] Missing expected features for ML prediction: {missing_features}. Available: {latest_features_series.index.tolist()[:10]}...")
                return None
            
            # Create a DataFrame with features in the correct order
            features_df_for_pred = pd.DataFrame([latest_features_series[self.feature_names_from_training]], columns=self.feature_names_from_training)
        except KeyError as e:
            self.logger.error(f"[{timestamp}] KeyError when selecting features for model: {e}. Expected: {self.feature_names_from_training[:10]}... Available: {latest_features_series.index.tolist()[:10]}...")
            return None
        except Exception as e:
            self.logger.error(f"[{timestamp}] Unexpected error preparing features for prediction: {e}", exc_info=True)
            return None

        if features_df_for_pred.isnull().any().any():
            self.logger.warning(f"[{timestamp}] NaNs found in features for ML prediction after alignment. Pipeline's imputer should handle this.")
            # self.logger.debug(f"Features with NaNs:\n{features_df_for_pred[features_df_for_pred.isnull().any(axis=1)]}")

        # Get prediction probabilities from the pipeline
        # Pipeline handles imputation and scaling internally.
        try:
            # predict_proba returns array of shape (n_samples, n_classes)
            # For binary (0=Loss, 1=Win), probabilities[0][1] is prob of Win (class 1)
            probabilities = self.ml_model_pipeline.predict_proba(features_df_for_pred)
            prob_class_1 = probabilities[0][1] # Assuming class 1 is "Win" or "Favorable Long"
            # prob_class_0 = probabilities[0][0] # Prob of "Loss" or "Favorable Short" (if binary)
        except Exception as e:
            self.logger.error(f"[{timestamp}] Error during ML model predict_proba: {e}", exc_info=True)
            self.logger.debug(f"Data passed to predict_proba (shape {features_df_for_pred.shape}):\n{features_df_for_pred.head().to_string()}")
            return None

        self.logger.info(f"[{timestamp}] ML Model predict_proba (for class 1 - 'Win'): {prob_class_1:.4f}")

        # Confirm based on candidate direction and configured thresholds
        # Project 1's labeling: 1=Win (long), 0=Loss (long).
        # So, prob_class_1 is the probability of a successful long trade.
        # For a SELL candidate, we'd ideally want a model that predicts short success,
        # or interpret low prob_class_1 as high prob_class_0 (short success).
        # Let's assume the model is trained to predict "favorable outcome for the candidate direction".
        # If the model is purely directional (predicts 1 for BUY, 0 for SELL), then prob_class_1 is P(BUY).
        # Given P1's labeling (win=1, loss=0 for a hypothetical long), prob_class_1 is P(Long Win).
        
        ml_confirms = False
        confirmation_probability = 0.0

        if candidate_direction == "BUY":
            required_threshold = self.config.ML_BUY_CONFIRM_THRESHOLD
            if prob_class_1 >= required_threshold:
                ml_confirms = True
                confirmation_probability = prob_class_1
        elif candidate_direction == "SELL":
            required_threshold = self.config.ML_SELL_CONFIRM_THRESHOLD
            # If model predicts P(Long Win), then for a SELL, we want P(Long Loss) to be high,
            # which means P(Long Win) should be low.
            # P(Long Loss) = probabilities[0][0] = 1 - prob_class_1
            prob_long_loss = 1.0 - prob_class_1
            if prob_long_loss >= required_threshold: # Threshold for P(Long Loss)
                ml_confirms = True
                confirmation_probability = prob_long_loss # This is prob of "short win" proxy
        else:
            self.logger.warning(f"[{timestamp}] Unknown candidate direction '{candidate_direction}' for ML confirmation.")
            return None

        if ml_confirms:
            self.logger.info(f"[{timestamp}] ML Confirmed Signal: {candidate_direction} with P={confirmation_probability:.4f} (Threshold: {required_threshold})")
            return {
                'signal': candidate_direction,
                'probability': confirmation_probability,
                'timestamp': timestamp
            }
        else:
            log_prob = prob_class_1 if candidate_direction == "BUY" else (1.0 - prob_class_1)
            self.logger.info(f"[{timestamp}] ML signal does not meet threshold for candidate '{candidate_direction}'. "
                             f"Relevant Prob={log_prob:.4f}, Required Threshold={required_threshold}")
            return None