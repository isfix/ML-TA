# ml_models/model_manager.py
"""
Manages machine learning models: Optuna hyperparameter tuning, training,
evaluation, ensembling, saving (joblib/json), loading, and prediction.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline 
from sklearn.impute import SimpleImputer

import joblib
import json
import optuna 

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None 

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    SMOTE = None 
    ImbPipeline = SklearnPipeline 

try:
    import config 
    from utilities import logging_utils, file_utils # Corrected: Direct import for flat structure
except ImportError:
    print("Warning: Could not perform standard config/utilities import in ModelManager. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger_mm = logging.getLogger(__name__) # Use different name to avoid clash with instance self.logger
    class MockConfigMM: LOG_FILE_APP = "model_manager_temp.log"
    if 'config' not in locals() and 'config' not in globals(): config_mm_fallback = MockConfigMM()
    # Fallback logger setup
    try: import logging_utils as temp_logging_utils_mm; logger_mm = temp_logging_utils_mm.setup_logger(__name__, getattr(config_mm_fallback, 'LOG_FILE_APP', "model_manager_temp.log")) # type: ignore
    except: pass # Basic logger_mm will be used
else: 
    logger_mm = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class ModelManager:
    def __init__(self, config_obj, logger_obj, model_name_prefix="", use_smote_cli=False):
        self.config = config_obj
        self.logger = logger_obj # Use passed logger for instance methods
        self.model_name_prefix = model_name_prefix 
        self.feature_names_ = None 
        self.scaler = None 
        self.use_smote = use_smote_cli and IMBLEARN_AVAILABLE and SMOTE is not None

        if self.use_smote and not IMBLEARN_AVAILABLE:
            self.logger.warning("SMOTE requested but imbalanced-learn library not installed. No SMOTE.")
            self.use_smote = False
        self.logger.info(f"ModelManager initialized. Prefix: '{self.model_name_prefix}'. SMOTE Active: {self.use_smote}")

    def _get_base_pipeline_steps(self, model_instance):
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        steps = [('imputer', imputer), ('scaler', scaler), ('classifier', model_instance)]
        return steps

    def _create_pipeline(self, model_instance, use_smote_for_this_pipeline=False, smote_k_neighbors=5):
        base_steps = self._get_base_pipeline_steps(model_instance)
        if use_smote_for_this_pipeline and self.use_smote: 
            smote_instance = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
            pipeline_steps = [base_steps[0]] + [('smote', smote_instance)] + base_steps[1:]
            pipeline = ImbPipeline(pipeline_steps)
            self.logger.info(f"Created ImbPipeline with SMOTE (k_neighbors={smote_k_neighbors}).")
        else:
            pipeline = SklearnPipeline(base_steps)
            self.logger.info("Created SklearnPipeline (SMOTE not active).")
        return pipeline

    def _get_model_instance(self, model_type_name, params=None):
        model_params = params or {} 
        if model_type_name == 'RandomForestClassifier':
            rf_defaults = {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced_subsample'}
            final_params = {**rf_defaults, **model_params}; return RandomForestClassifier(**final_params)
        elif model_type_name == 'XGBClassifier':
            if not XGBOOST_AVAILABLE: self.logger.error("XGBoost unavailable."); return None
            xgb_defaults = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42, 'use_label_encoder': False}
            final_params = {**xgb_defaults, **model_params}; return XGBClassifier(**final_params)
        else: self.logger.error(f"Unsupported model type: {model_type_name}"); return None

    def prepare_data_for_model(self, features_df_labeled: pd.DataFrame):
        self.logger.info("Preparing data for model training...")
        target_col = self.config.TARGET_COLUMN_NAME
        if target_col not in features_df_labeled.columns:
            self.logger.error(f"Target column '{target_col}' not found."); return [None]*7
        data_for_model = features_df_labeled[features_df_labeled[target_col].isin([0, 1])].copy()
        data_for_model.dropna(subset=[target_col], inplace=True)
        if data_for_model.empty: self.logger.error("No data after filtering for targets 0 or 1."); return [None]*7

        y = data_for_model[target_col].astype(int)
        X = data_for_model.drop(columns=[target_col, 'timestamp_ref_y'], errors='ignore') # Drop original target and any accidental timestamp_ref_y
        
        # Remove all columns that are purely objects or non-numeric before setting self.feature_names_
        X = X.select_dtypes(include=np.number)
        self.feature_names_ = X.columns.tolist()
        
        self.logger.info(f"Using {len(self.feature_names_)} numeric features: {self.feature_names_[:5]}...")
        if X.isnull().any().any(): self.logger.warning("NaNs in feature set (X) pre-split. Pipeline imputer will handle.")

        train_ratio = getattr(self.config, 'TRAIN_SPLIT_RATIO', 0.7)
        val_ratio = getattr(self.config, 'VALIDATION_SPLIT_RATIO', 0.15)
        train_size = int(len(X) * train_ratio); val_size = int(len(X) * val_ratio)
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_val, y_val = X.iloc[train_size : train_size + val_size], y.iloc[train_size : train_size + val_size]
        X_test, y_test = X.iloc[train_size + val_size :], y.iloc[train_size + val_size :]
        self.logger.info(f"Data split: Train ({len(X_train)}), Val ({len(X_val)}), Test ({len(X_test)})")
        if X_train.empty or X_val.empty: self.logger.error("Train/Val set empty after split."); return [None]*7
        
        # Scaler is part of pipeline, this self.scaler is mostly for compatibility if older logic needs it
        temp_scaler_for_return = StandardScaler()
        if not X_train.empty:
            temp_imputer = SimpleImputer(strategy='mean'); X_train_imputed = temp_imputer.fit_transform(X_train)
            temp_scaler_for_return.fit(X_train_imputed); self.scaler = temp_scaler_for_return
        else: self.scaler = None
        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler

    def optimize_hyperparameters(self, X_train_df, y_train_series, X_val_df, y_val_series, model_type_name):
        self.logger.info(f"Optuna optimization for {model_type_name}...")
        if X_val_df.empty or y_val_series.empty:
            self.logger.error("Validation set empty. Cannot run Optuna."); return self._get_model_instance(model_type_name).get_params(), -1

        def objective(trial):
            params = {}
            if model_type_name == 'RandomForestClassifier':
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
                params['max_depth'] = trial.suggest_int('max_depth', 5, 20) if trial.suggest_categorical('use_max_depth', [True, False]) else None
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 5)
                params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            elif model_type_name == 'XGBClassifier':
                if not XGBOOST_AVAILABLE: return 0.0 
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 8)
                params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
                params['gamma'] = trial.suggest_float('gamma', 0, 3)
            model_instance_optuna = self._get_model_instance(model_type_name, params)
            if model_instance_optuna is None: return 0.0 

            use_smote_trial = self.use_smote; smote_k_optuna = 1
            if use_smote_trial:
                min_class_count = y_train_series.value_counts().min()
                if min_class_count <= 1: use_smote_trial = False
                else: smote_k_optuna = max(1, min(5, min_class_count - 1))
            pipeline_optuna = self._create_pipeline(model_instance_optuna, use_smote_trial, smote_k_optuna)
            try:
                pipeline_optuna.fit(X_train_df, y_train_series); y_pred_val = pipeline_optuna.predict(X_val_df)
                metric = getattr(self.config, 'OPTUNA_OBJECTIVE_METRIC', 'f1_weighted')
                if metric == 'f1_weighted': score = f1_score(y_val_series, y_pred_val, average='weighted', zero_division=0)
                elif metric == 'accuracy': score = accuracy_score(y_val_series, y_pred_val)
                elif metric == 'roc_auc' and hasattr(pipeline_optuna, 'predict_proba'):
                    y_proba_val = pipeline_optuna.predict_proba(X_val_df)
                    score = roc_auc_score(y_val_series, y_proba_val[:, 1] if y_proba_val.shape[1]==2 else y_proba_val, multi_class='ovr', average='weighted')
                else: score = f1_score(y_val_series, y_pred_val, average='weighted', zero_division=0)
                return score
            except Exception as e: self.logger.error(f"Optuna trial error {model_type_name} params {params}: {e}", exc_info=False); return 0.0
        
        study = optuna.create_study(direction=getattr(self.config, 'OPTUNA_STUDY_DIRECTION', 'maximize'))
        try: study.optimize(objective, n_trials=getattr(self.config, 'OPTUNA_N_TRIALS', 10), timeout=getattr(self.config, 'OPTUNA_TIMEOUT_PER_STUDY', None), n_jobs=1) # n_jobs=-1 can cause issues with some models/loggers
        except Exception as e: self.logger.error(f"Optuna study failed for {model_type_name}: {e}", exc_info=True); return self._get_model_instance(model_type_name).get_params(), -1
        self.logger.info(f"Optuna for {model_type_name}: Best val={study.best_value:.4f}, Params={study.best_trial.params}")
        return study.best_trial.params, study.best_value

    def train_individual_model(self, X_train_df, y_train_series, X_val_df, y_val_series, model_type_name, model_params_override=None):
        self.logger.info(f"Training {model_type_name} {'with tuned params' if model_params_override else 'with defaults'}")
        model_instance = self._get_model_instance(model_type_name, model_params_override)
        if model_instance is None: return None
        use_smote_final = self.use_smote; smote_k_final = 1
        if use_smote_final:
            min_class_count = y_train_series.value_counts().min()
            if min_class_count <= 1: use_smote_final = False
            else: smote_k_final = max(1, min(5, min_class_count - 1))
        pipeline = self._create_pipeline(model_instance, use_smote_final, smote_k_final)
        try:
            pipeline.fit(X_train_df, y_train_series)
            self.logger.info(f"{model_type_name} pipeline trained.")
            if not X_val_df.empty and not y_val_series.empty:
                y_pred_val = pipeline.predict(X_val_df)
                self.logger.info(f"Validation Perf for {model_type_name}:\nAcc: {accuracy_score(y_val_series, y_pred_val):.4f}\n{classification_report(y_val_series, y_pred_val, zero_division=0)}\nCM:\n{confusion_matrix(y_val_series, y_pred_val)}")
            return pipeline
        except Exception as e: self.logger.error(f"Error training {model_type_name} pipeline: {e}", exc_info=True); return None

    def train_ensemble(self, X_train_df, y_train_series, X_val_df, y_val_series, tuned_base_model_pipelines_info: list):
        self.logger.info("Training ensemble model (VotingClassifier)...")
        if not tuned_base_model_pipelines_info: self.logger.error("No base pipelines for ensemble."); return None
        estimators = [(info['name'], info['pipeline']) for info in tuned_base_model_pipelines_info if 'pipeline' in info and 'name' in info]
        if not estimators: self.logger.error("No valid estimators for VotingClassifier."); return None
        ensemble_model = VotingClassifier(estimators=estimators, voting='soft') 
        try:
            ensemble_model.fit(X_train_df, y_train_series) 
            self.logger.info("Ensemble model 'fitted'.")
            if not X_val_df.empty and not y_val_series.empty:
                y_pred_val_ensemble = ensemble_model.predict(X_val_df)
                self.logger.info(f"Validation Perf for Ensemble:\nAcc: {accuracy_score(y_val_series, y_pred_val_ensemble):.4f}\n{classification_report(y_val_series, y_pred_val_ensemble, zero_division=0)}\nCM:\n{confusion_matrix(y_val_series, y_pred_val_ensemble)}")
            return ensemble_model
        except Exception as e: self.logger.error(f"Error training ensemble: {e}", exc_info=True); return None

    def save_model_pipeline(self, pipeline_to_save, full_model_name_prefix):
        model_filename = f"{full_model_name_prefix}_pipeline.joblib"
        feature_names_filename = f"{full_model_name_prefix}_feature_names.json"
        model_filepath = os.path.join(self.config.MODELS_DIR, model_filename)
        feature_names_filepath = os.path.join(self.config.MODELS_DIR, feature_names_filename)
        try:
            file_utils.ensure_dir(os.path.dirname(model_filepath)) 
            joblib.dump(pipeline_to_save, model_filepath)
            self.logger.info(f"Pipeline saved to {model_filepath}")
            if self.feature_names_:
                with open(feature_names_filepath, 'w') as f: json.dump(self.feature_names_, f)
                self.logger.info(f"Feature names saved to {feature_names_filepath}")
            else: self.logger.warning("Feature names not available to save.")
            return True
        except Exception as e: self.logger.error(f"Error saving model/features for {full_model_name_prefix}: {e}", exc_info=True); return False

    def load_model_pipeline(self, full_model_name_prefix):
        model_filename = f"{full_model_name_prefix}_pipeline.joblib"
        feature_names_filename = f"{full_model_name_prefix}_feature_names.json"
        model_filepath = os.path.join(self.config.MODELS_DIR, model_filename)
        feature_names_filepath = os.path.join(self.config.MODELS_DIR, feature_names_filename)
        pipeline_loaded, loaded_feature_names = None, None
        if not os.path.exists(model_filepath): self.logger.error(f"Model pipeline not found: {model_filepath}"); return None, None
        try:
            pipeline_loaded = joblib.load(model_filepath)
            self.logger.info(f"Pipeline loaded from {model_filepath}")
            if hasattr(pipeline_loaded, 'named_steps') and 'scaler' in pipeline_loaded.named_steps: self.scaler = pipeline_loaded.named_steps['scaler']
        except Exception as e: self.logger.error(f"Error loading pipeline from {model_filepath}: {e}", exc_info=True); return None, None
        if os.path.exists(feature_names_filepath):
            try:
                with open(feature_names_filepath, 'r') as f: loaded_feature_names = json.load(f)
                self.feature_names_ = loaded_feature_names 
                self.logger.info(f"Feature names loaded from {feature_names_filepath}")
            except Exception as e: self.logger.error(f"Error loading feature names from {feature_names_filepath}: {e}", exc_info=True)
        else: self.logger.warning(f"Feature names file not found: {feature_names_filepath}.")
        return pipeline_loaded, loaded_feature_names

    def predict(self, pipeline, X_live_df: pd.DataFrame, loaded_feature_names: list = None):
        if pipeline is None: self.logger.error("Pipeline not loaded."); return None
        if X_live_df.empty: self.logger.warning("Input DF for prediction empty."); return np.array([])
        features_to_use = loaded_feature_names or self.feature_names_
        if not features_to_use: self.logger.error("No feature names available for prediction."); return None
        try: X_aligned = X_live_df[features_to_use].copy()
        except KeyError as e: self.logger.error(f"Missing columns in prediction data: {set(features_to_use) - set(X_live_df.columns)}. Error: {e}"); return None
        try: return pipeline.predict(X_aligned)
        except Exception as e: self.logger.error(f"Error during prediction: {e}\nData head:\n{X_aligned.head().to_string()}", exc_info=True); return None

    def predict_proba(self, pipeline, X_live_df: pd.DataFrame, loaded_feature_names: list = None):
        if pipeline is None: self.logger.error("Pipeline not loaded."); return None
        if not hasattr(pipeline, 'predict_proba'): self.logger.error("Pipeline has no predict_proba."); return None
        if X_live_df.empty: self.logger.warning("Input DF for predict_proba empty."); return np.array([])
        features_to_use = loaded_feature_names or self.feature_names_
        if not features_to_use: self.logger.error("No feature names for predict_proba."); return None
        try: X_aligned = X_live_df[features_to_use].copy()
        except KeyError as e: self.logger.error(f"Missing columns in predict_proba data: {set(features_to_use) - set(X_live_df.columns)}. Error: {e}"); return None
        try: return pipeline.predict_proba(X_aligned)
        except Exception as e: self.logger.error(f"Error during predict_proba: {e}", exc_info=True); return None