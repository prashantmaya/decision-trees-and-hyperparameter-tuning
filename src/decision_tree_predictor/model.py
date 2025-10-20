"""Weather prediction model with Decision Tree and preprocessing pipeline."""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

logger = logging.getLogger(__name__)


class DecisionTreePredictor:
    """Weather prediction model using Decision Tree with preprocessing pipeline."""
    
    def __init__(
        self,
        max_depth: int = 7,
        max_leaf_nodes: int = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42
    ):
        """
        Initialize the Decision Tree weather predictor.
        
        Args:
            max_depth: Maximum depth of the tree
            max_leaf_nodes: Maximum number of leaf nodes
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random state for reproducibility
        """
        self.numerical_cols = [
            'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
            'Cloud3pm', 'Temp9am', 'Temp3pm'
        ]
        self.categorical_cols = [
            'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'
        ]
        self.target_col = 'RainTomorrow'
        
        # Initialize preprocessing components
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        
        # Initialize Decision Tree model with hyperparameters
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
        self.is_fitted = False
        self.metrics = {}
        self.feature_names = []
        self.hyperparameters = {
            'max_depth': max_depth,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
        
    def _validate_data(self, data: pd.DataFrame, is_training: bool = False):
        """Validate input data structure."""
        required_cols = self.numerical_cols + self.categorical_cols
        if is_training:
            required_cols.append(self.target_col)
        
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.debug(f"Data validation passed. Shape: {data.shape}")
        
    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None):
        """
        Fit the model and all preprocessing components.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset for metrics
        
        Returns:
            self
        """
        try:
            logger.info("Starting model training...")
            self._validate_data(train_data, is_training=True)
            
            # Prepare target
            y_train = train_data[self.target_col]
            logger.info(f"Training samples: {len(y_train)}, Positive class: {(y_train == 'Yes').sum()}")
            
            # Fit imputer and transform numerical columns
            logger.debug("Fitting imputer on numerical features...")
            self.imputer.fit(train_data[self.numerical_cols])
            X_numerical = self.imputer.transform(train_data[self.numerical_cols])
            
            # Fit scaler and transform numerical data
            logger.debug("Fitting scaler on numerical features...")
            self.scaler.fit(X_numerical)
            X_numerical_scaled = self.scaler.transform(X_numerical)
            
            # Fit encoder and transform categorical data
            logger.debug("Fitting encoder on categorical features...")
            X_categorical = train_data[self.categorical_cols].fillna("Unknown")
            self.encoder.fit(X_categorical)
            X_categorical_encoded = self.encoder.transform(X_categorical)
            
            # Store feature names
            encoded_features = list(self.encoder.get_feature_names_out(self.categorical_cols))
            self.feature_names = self.numerical_cols + encoded_features
            
            # Combine features
            X_train = np.hstack([X_numerical_scaled, X_categorical_encoded])
            logger.info(f"Final feature shape: {X_train.shape}")
            logger.info(f"Total features: {len(self.feature_names)}")
            
            # Fit the model
            logger.info("Training Decision Tree model...")
            logger.info(f"Hyperparameters: {self.hyperparameters}")
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_proba = self.model.predict_proba(X_train)[:, 1]
            
            self.metrics['train'] = {
                'accuracy': accuracy_score(y_train, train_predictions),
                'precision': precision_score(y_train, train_predictions, pos_label='Yes'),
                'recall': recall_score(y_train, train_predictions, pos_label='Yes'),
                'f1': f1_score(y_train, train_predictions, pos_label='Yes'),
                'roc_auc': roc_auc_score((y_train == 'Yes').astype(int), train_proba)
            }
            
            logger.info(f"Training metrics: {self.metrics['train']}")
            
            # Calculate validation metrics if provided
            if validation_data is not None:
                logger.info("Calculating validation metrics...")
                y_val = validation_data[self.target_col]
                X_val = self._preprocess(validation_data)
                val_predictions = self.model.predict(X_val)
                val_proba = self.model.predict_proba(X_val)[:, 1]
                
                self.metrics['validation'] = {
                    'accuracy': accuracy_score(y_val, val_predictions),
                    'precision': precision_score(y_val, val_predictions, pos_label='Yes'),
                    'recall': recall_score(y_val, val_predictions, pos_label='Yes'),
                    'f1': f1_score(y_val, val_predictions, pos_label='Yes'),
                    'roc_auc': roc_auc_score((y_val == 'Yes').astype(int), val_proba)
                }
                
                logger.info(f"Validation metrics: {self.metrics['validation']}")
            
            logger.info("Model training completed successfully!")
            return self
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            raise
    
    def _preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle numerical features
        X_numerical = self.imputer.transform(data[self.numerical_cols])
        X_numerical_scaled = self.scaler.transform(X_numerical)
        
        # Handle categorical features
        X_categorical = data[self.categorical_cols].fillna("Unknown")
        X_categorical_encoded = self.encoder.transform(X_categorical)
        
        # Combine features
        X = np.hstack([X_numerical_scaled, X_categorical_encoded])
        return X
    
    def predict(self, input_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Make a prediction for input data.
        
        Args:
            input_data: DataFrame with weather features
        
        Returns:
            Tuple of (prediction, probability)
        """
        try:
            self._validate_data(input_data, is_training=False)
            X = self._preprocess(input_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][list(self.model.classes_).index(prediction)]
            
            return prediction, float(probability)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        return importance_df
    
    def save(self, model_dir: str = "model"):
        """
        Save the model and all preprocessing components.
        
        Args:
            model_dir: Directory to save model files
        """
        try:
            model_path = Path(model_dir)
            model_path.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Saving model to {model_dir}...")
            
            # Save all components
            joblib.dump(self.imputer, model_path / "imputer.joblib")
            joblib.dump(self.scaler, model_path / "scaler.joblib")
            joblib.dump(self.encoder, model_path / "encoder.joblib")
            joblib.dump(self.model, model_path / "model.joblib")
            
            # Save metadata
            metadata = {
                'numerical_cols': self.numerical_cols,
                'categorical_cols': self.categorical_cols,
                'target_col': self.target_col,
                'is_fitted': self.is_fitted,
                'metrics': self.metrics,
                'feature_names': self.feature_names,
                'hyperparameters': self.hyperparameters
            }
            joblib.dump(metadata, model_path / "metadata.joblib")
            
            logger.info(f"Model saved successfully to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise
        
    @classmethod
    def load(cls, model_dir: str = "model") -> "DecisionTreePredictor":
        """
        Load a saved model and all preprocessing components.
        
        Args:
            model_dir: Directory containing model files
        
        Returns:
            Loaded DecisionTreePredictor instance
        """
        try:
            model_path = Path(model_dir)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            logger.info(f"Loading model from {model_dir}...")
            
            # Load metadata first to get hyperparameters
            metadata_file = model_path / "metadata.joblib"
            metadata = joblib.load(metadata_file) if metadata_file.exists() else {}
            
            # Initialize predictor with saved hyperparameters
            hyperparams = metadata.get('hyperparameters', {})
            predictor = cls(**hyperparams) if hyperparams else cls()
            
            # Load all components
            predictor.imputer = joblib.load(model_path / "imputer.joblib")
            predictor.scaler = joblib.load(model_path / "scaler.joblib")
            predictor.encoder = joblib.load(model_path / "encoder.joblib")
            predictor.model = joblib.load(model_path / "model.joblib")
            
            # Load metadata
            predictor.is_fitted = metadata.get('is_fitted', True)
            predictor.metrics = metadata.get('metrics', {})
            predictor.feature_names = metadata.get('feature_names', [])
            
            logger.info(f"Model loaded successfully from {model_dir}")
            logger.info(f"Model hyperparameters: {predictor.hyperparameters}")
            logger.info(f"Model metrics: {predictor.metrics}")
            
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

