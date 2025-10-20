"""Utility functions for the decision tree weather prediction system."""
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def format_prediction_response(prediction: str, probability: float) -> Dict[str, Any]:
    """
    Format prediction response.
    
    Args:
        prediction: The model prediction (Yes/No)
        probability: The prediction probability
    
    Returns:
        Formatted response dictionary
    """
    return {
        "will_rain_tomorrow": prediction == "Yes",
        "probability": round(probability, 4),
        "prediction_label": prediction
    }


def validate_weather_data(data: pd.DataFrame) -> bool:
    """
    Validate weather data for prediction.
    
    Args:
        data: Input DataFrame
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = [
        'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir',
        'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
        'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday'
    ]
    
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True


def log_prediction_request(location: str, features: Dict[str, Any]):
    """
    Log prediction request details.
    
    Args:
        location: Weather location
        features: Input features dictionary
    """
    logger.info(f"Prediction request - Location: {location}, Features: {len(features)} fields")


def calculate_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Create a formatted summary of model metrics.
    
    Args:
        metrics: Dictionary containing train and validation metrics
    
    Returns:
        Formatted metrics summary string
    """
    summary = []
    for dataset, values in metrics.items():
        summary.append(f"\n{dataset.capitalize()} Metrics:")
        for metric, value in values.items():
            summary.append(f"  {metric}: {value:.4f}")
    
    return "\n".join(summary)

