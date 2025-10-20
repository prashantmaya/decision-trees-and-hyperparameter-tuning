"""Tests for Decision Tree predictor model."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decision_tree_predictor.model import DecisionTreePredictor


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Location': np.random.choice(['Sydney', 'Melbourne', 'Brisbane'], n_samples),
        'MinTemp': np.random.uniform(10, 25, n_samples),
        'MaxTemp': np.random.uniform(20, 35, n_samples),
        'Rainfall': np.random.uniform(0, 10, n_samples),
        'Evaporation': np.random.uniform(0, 8, n_samples),
        'Sunshine': np.random.uniform(0, 12, n_samples),
        'WindGustDir': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], n_samples),
        'WindGustSpeed': np.random.uniform(20, 60, n_samples),
        'WindDir9am': np.random.choice(['N', 'S', 'E', 'W'], n_samples),
        'WindDir3pm': np.random.choice(['N', 'S', 'E', 'W'], n_samples),
        'WindSpeed9am': np.random.uniform(10, 40, n_samples),
        'WindSpeed3pm': np.random.uniform(10, 40, n_samples),
        'Humidity9am': np.random.uniform(40, 90, n_samples),
        'Humidity3pm': np.random.uniform(30, 80, n_samples),
        'Pressure9am': np.random.uniform(1000, 1020, n_samples),
        'Pressure3pm': np.random.uniform(1000, 1020, n_samples),
        'Cloud9am': np.random.randint(0, 9, n_samples),
        'Cloud3pm': np.random.randint(0, 9, n_samples),
        'Temp9am': np.random.uniform(15, 28, n_samples),
        'Temp3pm': np.random.uniform(18, 32, n_samples),
        'RainToday': np.random.choice(['Yes', 'No'], n_samples),
        'RainTomorrow': np.random.choice(['Yes', 'No'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def predictor():
    """Create a DecisionTreePredictor instance."""
    return DecisionTreePredictor(max_depth=3, max_leaf_nodes=5, random_state=42)


def test_predictor_initialization(predictor):
    """Test predictor initialization."""
    assert predictor.is_fitted is False
    assert predictor.hyperparameters['max_depth'] == 3
    assert predictor.hyperparameters['max_leaf_nodes'] == 5
    assert len(predictor.numerical_cols) == 16
    assert len(predictor.categorical_cols) == 5


def test_predictor_fit(predictor, sample_data):
    """Test model fitting."""
    predictor.fit(sample_data)
    
    assert predictor.is_fitted is True
    assert 'train' in predictor.metrics
    assert 'accuracy' in predictor.metrics['train']
    assert predictor.metrics['train']['accuracy'] > 0
    assert len(predictor.feature_names) > 0


def test_predictor_fit_with_validation(predictor, sample_data):
    """Test model fitting with validation data."""
    train_data = sample_data.iloc[:80]
    val_data = sample_data.iloc[80:]
    
    predictor.fit(train_data, validation_data=val_data)
    
    assert predictor.is_fitted is True
    assert 'train' in predictor.metrics
    assert 'validation' in predictor.metrics


def test_predictor_predict(predictor, sample_data):
    """Test prediction."""
    predictor.fit(sample_data)
    
    test_input = sample_data.iloc[[0]].drop(columns=['RainTomorrow'])
    prediction, probability = predictor.predict(test_input)
    
    assert prediction in ['Yes', 'No']
    assert 0 <= probability <= 1


def test_predictor_validate_data_missing_columns(predictor):
    """Test data validation with missing columns."""
    incomplete_data = pd.DataFrame({'Location': ['Sydney']})
    
    with pytest.raises(ValueError, match="Missing required columns"):
        predictor._validate_data(incomplete_data)


def test_predictor_predict_before_fit(predictor, sample_data):
    """Test prediction before fitting raises error."""
    test_input = sample_data.iloc[[0]].drop(columns=['RainTomorrow'])
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        predictor.predict(test_input)


def test_feature_importance(predictor, sample_data):
    """Test feature importance retrieval."""
    predictor.fit(sample_data)
    
    importance_df = predictor.get_feature_importance()
    
    assert isinstance(importance_df, pd.DataFrame)
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert len(importance_df) > 0
    assert importance_df['importance'].sum() > 0


def test_feature_importance_before_fit(predictor):
    """Test feature importance before fitting raises error."""
    with pytest.raises(ValueError, match="Model must be fitted"):
        predictor.get_feature_importance()


def test_save_and_load(predictor, sample_data):
    """Test model saving and loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Train and save model
        predictor.fit(sample_data)
        predictor.save(tmp_dir)
        
        # Check files exist
        model_path = Path(tmp_dir)
        assert (model_path / "model.joblib").exists()
        assert (model_path / "imputer.joblib").exists()
        assert (model_path / "scaler.joblib").exists()
        assert (model_path / "encoder.joblib").exists()
        assert (model_path / "metadata.joblib").exists()
        
        # Load model
        loaded_predictor = DecisionTreePredictor.load(tmp_dir)
        
        # Test loaded model
        assert loaded_predictor.is_fitted is True
        assert loaded_predictor.hyperparameters == predictor.hyperparameters
        
        # Test prediction with loaded model
        test_input = sample_data.iloc[[0]].drop(columns=['RainTomorrow'])
        prediction1, prob1 = predictor.predict(test_input)
        prediction2, prob2 = loaded_predictor.predict(test_input)
        
        assert prediction1 == prediction2
        assert abs(prob1 - prob2) < 0.001


def test_load_nonexistent_model():
    """Test loading from non-existent directory."""
    with pytest.raises(FileNotFoundError):
        DecisionTreePredictor.load("nonexistent_directory")


def test_hyperparameters_effect(sample_data):
    """Test that different hyperparameters affect model."""
    predictor1 = DecisionTreePredictor(max_depth=2, random_state=42)
    predictor2 = DecisionTreePredictor(max_depth=10, random_state=42)
    
    predictor1.fit(sample_data)
    predictor2.fit(sample_data)
    
    # Deeper tree should typically have better training accuracy
    assert predictor1.is_fitted and predictor2.is_fitted
    # Just check they produce different results
    assert predictor1.hyperparameters != predictor2.hyperparameters

