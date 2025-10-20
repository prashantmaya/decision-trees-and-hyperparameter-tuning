"""FastAPI application for weather prediction using Decision Tree."""
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import pandas as pd

from .model import DecisionTreePredictor
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE) if Path(settings.LOG_FILE).parent.exists() or Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True) else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global model instance
model: DecisionTreePredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting up application...")
    global model
    try:
        model = DecisionTreePredictor.load(settings.MODEL_DIR)
        logger.info(f"Model loaded successfully from {settings.MODEL_DIR}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready weather prediction system using Decision Tree with hyperparameter tuning",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class WeatherInput(BaseModel):
    """Input schema for weather prediction."""
    Location: str = Field(..., description="Weather station location")
    MinTemp: float = Field(..., description="Minimum temperature in celsius", ge=-50, le=60)
    MaxTemp: float = Field(..., description="Maximum temperature in celsius", ge=-50, le=60)
    Rainfall: float = Field(..., description="Rainfall in mm", ge=0)
    Evaporation: Optional[float] = Field(None, description="Evaporation in mm", ge=0)
    Sunshine: Optional[float] = Field(None, description="Sunshine hours", ge=0, le=24)
    WindGustDir: str = Field(..., description="Wind gust direction")
    WindGustSpeed: float = Field(..., description="Wind gust speed in km/h", ge=0)
    WindDir9am: str = Field(..., description="Wind direction at 9am")
    WindDir3pm: str = Field(..., description="Wind direction at 3pm")
    WindSpeed9am: float = Field(..., description="Wind speed at 9am in km/h", ge=0)
    WindSpeed3pm: float = Field(..., description="Wind speed at 3pm in km/h", ge=0)
    Humidity9am: float = Field(..., description="Humidity at 9am", ge=0, le=100)
    Humidity3pm: float = Field(..., description="Humidity at 3pm", ge=0, le=100)
    Pressure9am: float = Field(..., description="Atmospheric pressure at 9am in hPa", ge=900, le=1100)
    Pressure3pm: float = Field(..., description="Atmospheric pressure at 3pm in hPa", ge=900, le=1100)
    Cloud9am: float = Field(..., description="Cloud cover at 9am (oktas)", ge=0, le=9)
    Cloud3pm: float = Field(..., description="Cloud cover at 3pm (oktas)", ge=0, le=9)
    Temp9am: float = Field(..., description="Temperature at 9am in celsius", ge=-50, le=60)
    Temp3pm: float = Field(..., description="Temperature at 3pm in celsius", ge=-50, le=60)
    RainToday: str = Field(..., description="Did it rain today? (Yes/No)")
    
    @field_validator('RainToday')
    @classmethod
    def validate_rain_today(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('RainToday must be either "Yes" or "No"')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Location": "Sydney",
                "MinTemp": 18.0,
                "MaxTemp": 25.0,
                "Rainfall": 0.0,
                "Evaporation": 4.0,
                "Sunshine": 8.0,
                "WindGustDir": "NE",
                "WindGustSpeed": 35.0,
                "WindDir9am": "N",
                "WindDir3pm": "NE",
                "WindSpeed9am": 15.0,
                "WindSpeed3pm": 20.0,
                "Humidity9am": 70.0,
                "Humidity3pm": 60.0,
                "Pressure9am": 1015.0,
                "Pressure3pm": 1013.0,
                "Cloud9am": 5.0,
                "Cloud3pm": 4.0,
                "Temp9am": 20.0,
                "Temp3pm": 24.0,
                "RainToday": "No"
            }
        }


class WeatherPrediction(BaseModel):
    """Output schema for weather prediction."""
    will_rain_tomorrow: bool = Field(..., description="Will it rain tomorrow?")
    probability: float = Field(..., description="Confidence probability", ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    timestamp: datetime


class FeatureImportance(BaseModel):
    """Feature importance response."""
    feature: str
    importance: float


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "model_type": "Decision Tree Classifier",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        version=settings.APP_VERSION,
        model_loaded=model is not None,
        timestamp=datetime.utcnow()
    )


@app.post(f"{settings.API_PREFIX}/predict", response_model=WeatherPrediction, tags=["Prediction"])
async def predict_weather(weather_input: WeatherInput):
    """
    Predict whether it will rain tomorrow based on today's weather conditions.
    
    Uses a Decision Tree Classifier with optimized hyperparameters:
    - max_depth: Controls tree depth to prevent overfitting
    - max_leaf_nodes: Limits the number of leaf nodes
    - min_samples_split: Minimum samples to split a node
    - min_samples_leaf: Minimum samples at leaf nodes
    
    Returns the prediction along with the confidence probability.
    """
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        logger.info(f"Received prediction request for location: {weather_input.Location}")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([weather_input.model_dump()])
        
        # Make prediction
        prediction, probability = model.predict(input_df)
        
        result = WeatherPrediction(
            will_rain_tomorrow=(prediction == "Yes"),
            probability=float(probability)
        )
        
        logger.info(f"Prediction successful: {result.will_rain_tomorrow} (probability: {result.probability:.2f})")
        return result
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get(f"{settings.API_PREFIX}/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    return {
        "model_type": "Decision Tree Classifier",
        "hyperparameters": model.hyperparameters,
        "numerical_features": model.numerical_cols,
        "categorical_features": model.categorical_cols,
        "target": model.target_col,
        "metrics": model.metrics
    }


@app.get(f"{settings.API_PREFIX}/model/feature-importance", tags=["Model"])
async def get_feature_importance():
    """
    Get feature importance from the trained Decision Tree model.
    
    Returns features sorted by importance score (higher is more important).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        importance_df = model.get_feature_importance()
        
        # Convert to list of dicts for JSON response
        features = [
            {"feature": row["feature"], "importance": float(row["importance"])}
            for _, row in importance_df.head(20).iterrows()  # Top 20 features
        ]
        
        return {
            "total_features": len(importance_df),
            "top_features": features
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

