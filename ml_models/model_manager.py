# ml_models/model_manager.py
"""
Manages machine learning models: Optuna hyperparameter tuning, training,
evaluation, ensembling, saving (joblib/json), loading, and prediction.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit # TimeSeriesSplit for robust validation
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline # Renamed to avoid clash
from sklearn.impute import SimpleImputer

import joblib
import json
import optuna # For hyperparameter optimization

# Conditional import for XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None # Define it as None if not available

# Conditional import for SMOTE (from imblearn)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    SMOTE = None # Define as None
    ImbPipeline = SklearnPipeline # Fallback to scikit-learn's Pipeline if imblearn is not available

# Import config and logging_utils (assuming they are accessible)
# This might require adjusting sys.path if run directly or if project structure is deep
try:
    import config # Assuming config.py is in the parent directory or accessible
    from utilities import logging_utils, file_utils # Assuming utilities is a sibling package or accessible
except ImportError:
    # Fallback for direct execution or specific project structures
    # This is a common pattern if your main script handles path adjustments
    import sys
    # Example: if model_manager.py is in trading_workflow/ml_models/
    # and config.py is in trading_workflow/
    # and utilities/ is in trading_workflow/
    # then config and utilities should be directly importable if trading_workflow is the root for execution.
    # If running model_manager.py directly, you might need:
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir) # Up one level to ml_models/
    # project_root = os.path.dirname(project_root) # Up another to trading_workflow/
    # sys.path.insert(0, project_root)
    # import config
    # from utilities import logging_utils, file_utils
    print("Warning: Could not perform standard config/utilities import in ModelManager. Ensure paths are correct.")
    # For now, assume config and logging_utils will be available when called from main.py
    pass


class ModelManager:
    def __init__(self, config_obj, logger_obj, model_name_prefix="", use_smote_cli=False):
        self.config = config_obj
        self.logger = logger_obj
        self.model_name_prefix = model_name_prefix # e.g., "EURUSD_M5" - used for saving/loading specific models
        self.feature_names_ = None # To store feature names after preparation
        self.scaler = None # Will be part of the pipeline, but can be stored separately if needed after fit
        self.use_smote = use_smote_cli and IMBLEARN_AVAILABLE and SMOTE is not None

        if self.use_smote and not IMBLEARN_AVAILABLE:
            self.logger.warning("SMOTE requested but imbalanced-learn library not installed. Proceeding without SMOTE.")
            self.use_smote = False

        self.logger.info(f"ModelManager initialized. Prefix: '{self.model_name_prefix}'. SMOTE (CLI): {use_smote_cli}, SMOTE (Active): {self.use_smote}")

    def _get_base_pipeline_steps(self, model_instance):
        """Returns a list of base pipeline steps (imputer, scaler, classifier)."""
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        steps = [
            ('imputer', imputer),
            ('scaler', scaler),
            ('classifier', model_instance)
        ]
        return steps

    def _create_pipeline(self, model_instance, use_smote_for_this_pipeline=False, smote_k_neighbors=5):
        """Creates a scikit-learn or imblearn pipeline."""
        base_steps = self._get_base_pipeline_steps(model_instance)
        if use_smote_for_this_pipeline and self.use_smote: # Check both global self.use_smote and local flag
            smote_instance = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
            # Insert SMOTE before scaler
            pipeline_steps = [base_steps[0]] + [('smote', smote_instance)] + base_steps[1:]
            pipeline = ImbPipeline(pipeline_steps)
            self.logger.info(f"Created ImbPipeline with SMOTE (k_neighbors={smote_k_neighbors}).")
        else:
            pipeline = SklearnPipeline(base_steps)
            self.logger.info("Created SklearnPipeline (SMOTE not active for this pipeline).")
        return pipeline

    def _get_model_instance(self, model_type_name, params=None):
        """Instantiates a model based on its type name and parameters."""
        model_params = params or {} # Use provided params or empty dict for defaults
        if model_type_name == 'RandomForestClassifier':
            # Default params for RF if not provided
            rf_defaults = {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced_subsample'}
            final_params = {**rf_defaults, **model_params}
            return RandomForestClassifier(**final_params)
        elif model_type_name == 'XGBClassifier':
            if not XGBOOST_AVAILABLE:
                self.logger.error("XGBoost is not available. Cannot create XGBClassifier instance.")
                return None
            # Default params for XGB if not provided
            xgb_defaults = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42, 'use_label_encoder': False}
            final_params = {**xgb_defaults, **model_params}
            return XGBClassifier(**final_params)
        else:
            self.logger.error(f"Unsupported model type: {model_type_name}")
            return None

    def prepare_data_for_model(self, features_df_labeled: pd.DataFrame):
        self.logger.info("Preparing data for model training...")
        target_col = self.config.TARGET_COLUMN_NAME
        if target_col not in features_df_labeled.columns:
            self.logger.error(f"Target column '{target_col}' not found. Cannot prepare data."); return [None]*7

        # Filter for valid targets (0 or 1, assuming binary classification for now)
        # Project 1's label_signals creates 0 (loss), 1 (win), 2 (neutral)
        data_for_model = features_df_labeled[features_df_labeled[target_col].isin([0, 1])].copy()
        data_for_model.dropna(subset=[target_col], inplace=True)

        if data_for_model.empty:
            self.logger.error("No data remaining after filtering for valid targets (0 or 1)."); return [None]*7

        y = data_for_model[target_col].astype(int)
        X = data_for_model.drop(columns=[target_col])
        self.feature_names_ = X.columns.tolist() # Store feature names before any transformation
        self.logger.info(f"Using {len(self.feature_names_)} features: {self.feature_names_[:5]}...")

        X = X.apply(pd.to_numeric, errors='coerce') # Ensure all features are numeric
        if X.isnull().any().any():
            self.logger.warning("NaNs found in feature set (X) before split. Imputation will be handled by pipeline.")
            # No fillna here; imputer in pipeline will handle it based on training data.

        # Time-series aware split (e.g., 70% train, 15% val, 15% test)
        train_ratio = getattr(self.config, 'TRAIN_SPLIT_RATIO', 0.7)
        val_ratio = getattr(self.config, 'VALIDATION_SPLIT_RATIO', 0.15)
        # test_ratio = 1.0 - train_ratio - val_ratio # Implicit

        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)

        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_val, y_val = X.iloc[train_size : train_size + val_size], y.iloc[train_size : train_size + val_size]
        X_test, y_test = X.iloc[train_size + val_size :], y.iloc[train_size + val_size :]

        self.logger.info(f"Data split: Train ({len(X_train)}), Validation ({len(X_val)}), Test ({len(X_test)})")

        if X_train.empty or X_val.empty:
            self.logger.error("Training or validation set is empty after split. Not enough data."); return [None]*7

        # Scaler will be part of the pipeline, fitted on X_train during pipeline.fit()
        # For now, we return unscaled data. The pipeline handles scaling.
        # We need to save the fitted scaler from the pipeline later.
        # For consistency with P1, let's fit a scaler here to return, but the pipeline will have its own.
        temp_scaler_for_return = StandardScaler()
        if not X_train.empty:
             # Impute NaNs before fitting scaler for this standalone scaler instance
            temp_imputer = SimpleImputer(strategy='mean')
            X_train_imputed_for_scaler = temp_imputer.fit_transform(X_train)
            temp_scaler_for_return.fit(X_train_imputed_for_scaler)
            self.scaler = temp_scaler_for_return # Store this instance if needed by P1's original flow
            self.logger.info("Standalone scaler fitted on (imputed) training data for potential separate use.")
        else:
            self.scaler = None

        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler # Return unscaled X for pipeline

    def optimize_hyperparameters(self, X_train_df, y_train_series, X_val_df, y_val_series, model_type_name):
        self.logger.info(f"Starting Optuna hyperparameter optimization for {model_type_name}...")
        
        # Data for Optuna objective function (should be unscaled, pipeline handles it)
        # Optuna needs to evaluate on validation set
        if X_val_df.empty or y_val_series.empty:
            self.logger.error("Validation set is empty. Cannot run Optuna. Returning default params.")
            return self._get_model_instance(model_type_name).get_params(), -1 # Return default params and a bad score

        def objective(trial):
            params = {}
            if model_type_name == 'RandomForestClassifier':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['max_depth'] = trial.suggest_int('max_depth', 5, 30, log=True) # Allow None by not suggesting if condition met
                if trial.suggest_categorical('use_max_depth', [True, False]): # Control if max_depth is used
                     pass # max_depth already suggested
                else:
                    params['max_depth'] = None # Explicitly set to None
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
                params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            elif model_type_name == 'XGBClassifier':
                if not XGBOOST_AVAILABLE: return 0.0 # Should not happen if checked before
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 400)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
                params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                params['gamma'] = trial.suggest_float('gamma', 0, 5)
                # Scale pos weight for imbalance if not using SMOTE in this specific trial pipeline
                # This is complex if SMOTE is also in the pipeline. For now, assume SMOTE handles imbalance.
            else:
                self.logger.warning(f"No Optuna hyperparameter suggestions defined for {model_type_name}. Using defaults.")

            model_instance_optuna = self._get_model_instance(model_type_name, params)
            if model_instance_optuna is None: return 0.0 # Score for failure

            # Determine SMOTE k_neighbors for this trial based on y_train_series
            use_smote_for_trial = self.use_smote # Use the global SMOTE setting from ModelManager init
            smote_k_optuna = 1
            if use_smote_for_trial:
                min_class_count_train = y_train_series.value_counts().min()
                if min_class_count_train <= 1: use_smote_for_trial = False # Cannot use SMOTE
                else: smote_k_optuna = max(1, min(5, min_class_count_train - 1))
            
            pipeline_optuna = self._create_pipeline(model_instance_optuna, use_smote_for_trial, smote_k_optuna)
            
            try:
                pipeline_optuna.fit(X_train_df, y_train_series)
                y_pred_val = pipeline_optuna.predict(X_val_df)
                
                # Choose metric for Optuna (e.g., F1-weighted, ROC-AUC if binary/proba)
                metric_choice = getattr(self.config, 'OPTUNA_OBJECTIVE_METRIC', 'f1_weighted')
                if metric_choice == 'f1_weighted':
                    score = f1_score(y_val_series, y_pred_val, average='weighted', zero_division=0)
                elif metric_choice == 'accuracy':
                    score = accuracy_score(y_val_series, y_pred_val)
                elif metric_choice == 'roc_auc' and hasattr(pipeline_optuna, 'predict_proba'):
                    y_proba_val = pipeline_optuna.predict_proba(X_val_df)
                    if y_proba_val.shape[1] == 2: # Binary classification
                        score = roc_auc_score(y_val_series, y_proba_val[:, 1])
                    else: # Multiclass (One-vs-Rest)
                        score = roc_auc_score(y_val_series, y_proba_val, multi_class='ovr', average='weighted')
                else: # Default to F1 weighted
                    self.logger.warning(f"Unsupported or inapplicable Optuna metric '{metric_choice}'. Defaulting to f1_weighted.")
                    score = f1_score(y_val_series, y_pred_val, average='weighted', zero_division=0)
                return score
            except Exception as e:
                self.logger.error(f"Error during Optuna trial for {model_type_name} with params {params}: {e}", exc_info=False) # exc_info=False to reduce log spam
                return 0.0 # Return a low score if trial fails

        study_direction = getattr(self.config, 'OPTUNA_STUDY_DIRECTION', 'maximize')
        study = optuna.create_study(direction=study_direction)
        n_trials = getattr(self.config, 'OPTUNA_N_TRIALS', 25)
        timeout = getattr(self.config, 'OPTUNA_TIMEOUT_PER_STUDY', None)
        
        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=-1) # n_jobs=-1 for parallel if objective is thread-safe
        except Exception as e:
            self.logger.error(f"Optuna study optimization failed for {model_type_name}: {e}", exc_info=True)
            return self._get_model_instance(model_type_name).get_params(), -1 # Default params, bad score

        self.logger.info(f"Optuna study for {model_type_name} complete. Best value: {study.best_value:.4f}")
        self.logger.info(f"Best hyperparameters: {study.best_trial.params}")
        return study.best_trial.params, study.best_value

    def train_individual_model(self, X_train_df, y_train_series, X_val_df, y_val_series, model_type_name, model_params_override=None):
        """Trains an individual model, possibly with overridden (tuned) parameters."""
        self.logger.info(f"Training individual model: {model_type_name} {'with overridden params' if model_params_override else 'with default params'}")
        
        model_instance = self._get_model_instance(model_type_name, model_params_override)
        if model_instance is None: return None

        # Determine SMOTE k_neighbors for final training based on y_train_series
        use_smote_final = self.use_smote
        smote_k_final = 1
        if use_smote_final:
            min_class_count_train = y_train_series.value_counts().min()
            if min_class_count_train <= 1: use_smote_final = False
            else: smote_k_final = max(1, min(5, min_class_count_train - 1))

        pipeline = self._create_pipeline(model_instance, use_smote_final, smote_k_final)

        try:
            # For final model, often good to train on combined train+validation data if Optuna used validation for tuning
            # X_train_full = pd.concat([X_train_df, X_val_df]) if not X_val_df.empty else X_train_df
            # y_train_full = pd.concat([y_train_series, y_val_series]) if not y_val_series.empty else y_train_series
            # pipeline.fit(X_train_full, y_train_full)
            # OR, stick to just training data for simplicity and use validation purely for tuning/early stopping idea
            pipeline.fit(X_train_df, y_train_series)
            self.logger.info(f"{model_type_name} model pipeline trained successfully.")

            # Evaluate on validation set (if provided and not empty)
            if not X_val_df.empty and not y_val_series.empty:
                y_pred_val = pipeline.predict(X_val_df)
                self.logger.info(f"\n--- Validation Set Performance for {model_type_name} ---")
                self.logger.info(f"Accuracy: {accuracy_score(y_val_series, y_pred_val):.4f}")
                self.logger.info(f"\n{classification_report(y_val_series, y_pred_val, zero_division=0)}")
                self.logger.info(f"Confusion Matrix:\n{confusion_matrix(y_val_series, y_pred_val)}")
            else:
                self.logger.info("Validation set empty or not provided, skipping validation metrics for this training run.")
            return pipeline # Return the fitted pipeline
        except Exception as e:
            self.logger.error(f"Error training {model_type_name} pipeline: {e}", exc_info=True)
            return None

    def train_ensemble(self, X_train_df, y_train_series, X_val_df, y_val_series, tuned_base_model_pipelines: list):
        """ Trains an ensemble model (VotingClassifier) using pre-trained (tuned) base model pipelines. """
        self.logger.info("Training ensemble model (VotingClassifier)...")
        if not tuned_base_model_pipelines:
            self.logger.error("No base model pipelines provided for ensemble."); return None

        # Estimators for VotingClassifier should be (name, model_pipeline)
        estimators = []
        for i, pipeline_info in enumerate(tuned_base_model_pipelines):
            # pipeline_info could be a dict {'name': 'RF', 'pipeline': fitted_rf_pipeline} or just the pipeline
            if isinstance(pipeline_info, dict) and 'name' in pipeline_info and 'pipeline' in pipeline_info:
                 estimators.append((f"{pipeline_info['name']}_{i}", pipeline_info['pipeline']))
            elif hasattr(pipeline_info, 'fit') and hasattr(pipeline_info, 'predict'): # It's a pipeline object
                 estimators.append((f"model_{i}", pipeline_info))
            else:
                self.logger.warning(f"Invalid item in tuned_base_model_pipelines: {pipeline_info}. Skipping.")
        
        if not estimators: self.logger.error("No valid estimators for VotingClassifier."); return None

        ensemble_model = VotingClassifier(estimators=estimators, voting='soft') # Soft voting uses predict_proba

        try:
            # Fit the ensemble model. It uses the pre-fitted base pipelines for prediction.
            # The VotingClassifier itself is "fitted" by learning how to combine these.
            # It doesn't re-train the base models.
            ensemble_model.fit(X_train_df, y_train_series) # Needs X, y to fit the voter
            self.logger.info("Ensemble model (VotingClassifier) 'fitted' successfully.")

            if not X_val_df.empty and not y_val_series.empty:
                y_pred_val_ensemble = ensemble_model.predict(X_val_df)
                self.logger.info(f"\n--- Validation Set Performance for Ensemble ---")
                self.logger.info(f"Accuracy: {accuracy_score(y_val_series, y_pred_val_ensemble):.4f}")
                self.logger.info(f"\n{classification_report(y_val_series, y_pred_val_ensemble, zero_division=0)}")
                self.logger.info(f"Confusion Matrix:\n{confusion_matrix(y_val_series, y_pred_val_ensemble)}")
            return ensemble_model
        except Exception as e:
            self.logger.error(f"Error training ensemble model: {e}", exc_info=True)
            return None

    def save_model_pipeline(self, pipeline_to_save, full_model_name_prefix):
        """Saves the model pipeline, scaler (from pipeline), and feature names."""
        # full_model_name_prefix already includes pair, model type, and _tuned or _ensemble
        model_filename = f"{full_model_name_prefix}_pipeline.joblib"
        # Scaler is part of the pipeline, but P1 saved it separately.
        # We can extract it if needed, or just save the whole pipeline.
        # For simplicity, we save the whole pipeline. The scaler can be accessed via pipeline.named_steps['scaler']
        feature_names_filename = f"{full_model_name_prefix}_feature_names.json"

        model_filepath = os.path.join(self.config.MODELS_DIR, model_filename)
        feature_names_filepath = os.path.join(self.config.MODELS_DIR, feature_names_filename)

        try:
            file_utils.ensure_dir(os.path.dirname(model_filepath)) # From P2 file_utils
            joblib.dump(pipeline_to_save, model_filepath)
            self.logger.info(f"Model pipeline successfully saved to {model_filepath}")

            # Save feature names (should be stored in self.feature_names_ during prepare_data)
            if self.feature_names_:
                with open(feature_names_filepath, 'w') as f:
                    json.dump(self.feature_names_, f)
                self.logger.info(f"Feature names successfully saved to {feature_names_filepath}")
            else:
                self.logger.warning("Feature names not available in ModelManager. Not saving feature names file.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model pipeline/features for prefix {full_model_name_prefix}: {e}", exc_info=True)
            return False

    def load_model_pipeline(self, full_model_name_prefix):
        """Loads a model pipeline and its associated feature names."""
        model_filename = f"{full_model_name_prefix}_pipeline.joblib"
        feature_names_filename = f"{full_model_name_prefix}_feature_names.json"
        model_filepath = os.path.join(self.config.MODELS_DIR, model_filename)
        feature_names_filepath = os.path.join(self.config.MODELS_DIR, feature_names_filename)

        pipeline_loaded = None
        loaded_feature_names = None

        if not os.path.exists(model_filepath):
            self.logger.error(f"Model pipeline file not found: {model_filepath}"); return None, None
        try:
            pipeline_loaded = joblib.load(model_filepath)
            self.logger.info(f"Model pipeline successfully loaded from {model_filepath}")
            # Extract scaler for compatibility if P1 logic expects separate scaler
            if hasattr(pipeline_loaded, 'named_steps') and 'scaler' in pipeline_loaded.named_steps:
                self.scaler = pipeline_loaded.named_steps['scaler']
        except Exception as e:
            self.logger.error(f"Error loading model pipeline from {model_filepath}: {e}", exc_info=True); return None, None

        if not os.path.exists(feature_names_filepath):
            self.logger.warning(f"Feature names file not found: {feature_names_filepath}. Features may not align.")
        else:
            try:
                with open(feature_names_filepath, 'r') as f:
                    loaded_feature_names = json.load(f)
                self.feature_names_ = loaded_feature_names # Store them in the instance
                self.logger.info(f"Feature names successfully loaded from {feature_names_filepath}")
            except Exception as e:
                self.logger.error(f"Error loading feature names from {feature_names_filepath}: {e}", exc_info=True)
                # Continue without feature names, but log error

        return pipeline_loaded, loaded_feature_names # Return pipeline and feature names

    def predict(self, pipeline, X_live_df: pd.DataFrame, loaded_feature_names: list = None):
        """Makes predictions using the loaded pipeline. Ensures feature alignment."""
        if pipeline is None: self.logger.error("Pipeline not loaded. Cannot predict."); return None
        if X_live_df.empty: self.logger.warning("Input DataFrame for prediction is empty."); return np.array([])

        # Ensure features are in the correct order and only expected features are present
        features_to_use = loaded_feature_names or self.feature_names_
        if not features_to_use:
            self.logger.error("No feature names available (neither passed nor stored). Cannot reliably predict.")
            # As a last resort, try with all columns in X_live_df, but this is risky.
            # For now, let's require feature names for safety.
            return None
        
        try:
            # Reorder/select columns in X_live_df to match training feature order
            X_aligned = X_live_df[features_to_use].copy()
        except KeyError as e:
            missing_cols = set(features_to_use) - set(X_live_df.columns)
            self.logger.error(f"Missing expected feature columns in prediction data: {missing_cols}. Error: {e}")
            return None
        
        try:
            predictions = pipeline.predict(X_aligned)
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)
            self.logger.debug(f"Data passed to predict (shape {X_aligned.shape}):\n{X_aligned.head().to_string()}")
            return None

    def predict_proba(self, pipeline, X_live_df: pd.DataFrame, loaded_feature_names: list = None):
        """Makes probability predictions. Ensures feature alignment."""
        if pipeline is None: self.logger.error("Pipeline not loaded. Cannot predict_proba."); return None
        if not hasattr(pipeline, 'predict_proba'):
            self.logger.error("Loaded pipeline does not support predict_proba."); return None
        if X_live_df.empty: self.logger.warning("Input DataFrame for predict_proba is empty."); return np.array([])

        features_to_use = loaded_feature_names or self.feature_names_
        if not features_to_use:
            self.logger.error("No feature names available for predict_proba. Cannot reliably predict."); return None
        
        try:
            X_aligned = X_live_df[features_to_use].copy()
        except KeyError as e:
            missing_cols = set(features_to_use) - set(X_live_df.columns)
            self.logger.error(f"Missing expected feature columns in predict_proba data: {missing_cols}. Error: {e}")
            return None

        try:
            probabilities = pipeline.predict_proba(X_aligned)
            return probabilities
        except Exception as e:
            self.logger.error(f"Error during predict_proba: {e}", exc_info=True)
            return None