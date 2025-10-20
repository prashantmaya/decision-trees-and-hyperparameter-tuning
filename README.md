# 🌦️ Decision Tree Weather Prediction System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E.svg)](https://scikit-learn.org/)

A production-ready weather prediction system that uses a Decision Tree Classifier with optimized hyperparameters to predict whether it will rain tomorrow based on today's weather conditions. Built with FastAPI, scikit-learn, and Docker.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Details](#model-details)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Production-Ready API**: FastAPI-based REST API with automatic OpenAPI documentation
- **Optimized Decision Tree**: Hyperparameter-tuned model for better performance
- **Complete ML Pipeline**: Preprocessing, training, and inference pipeline
- **Model Persistence**: Save and load trained models with all preprocessing components
- **Feature Importance**: Analyze which features matter most for predictions
- **Input Validation**: Comprehensive validation with Pydantic
- **Structured Logging**: Detailed logging system for monitoring and debugging
- **Comprehensive Testing**: Unit and integration tests with pytest
- **Docker Support**: Fully containerized application
- **Health Checks**: Monitoring and health check endpoints
- **Configuration Management**: Environment-based configuration with sensible defaults
- **Error Handling**: Robust error handling and validation

## 🛠️ Tech Stack

- **Python 3.11+**
- **FastAPI** - Modern, fast web framework for APIs
- **scikit-learn** - Machine learning library
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Pydantic** - Data validation using Python type annotations
- **Docker & Docker Compose** - Containerization
- **pytest** - Testing framework
- **Uvicorn** - ASGI server

## 🌲 Model Details

### Decision Tree Classifier

This project uses a Decision Tree Classifier with carefully tuned hyperparameters:

- **max_depth**: 7 - Controls tree depth to prevent overfitting
- **max_leaf_nodes**: 8 - Limits the number of leaf nodes
- **min_samples_split**: 2 - Minimum samples required to split a node
- **min_samples_leaf**: 1 - Minimum samples required at a leaf node

### Hyperparameter Tuning

The hyperparameters were optimized using:

- Grid Search with cross-validation
- Validation set performance monitoring
- Analysis of training vs validation accuracy curves

### Preprocessing Pipeline

1. **Numerical Features**: Mean imputation → Min-Max scaling
2. **Categorical Features**: Unknown value filling → One-Hot encoding
3. **Feature Engineering**: 16 numerical + 5 categorical features

### Model Performance

The model achieves:

- **Training Accuracy**: ~83-85%
- **Validation Accuracy**: ~84%
- **ROC-AUC**: ~0.85-0.90

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/prashantmaya/decision-trees-and-hyperparameter-tuning.git

cd decision-tress-and-hyperparameters

# Build and run with Docker Compose
docker-compose up -d

# Check if the service is running
curl http://localhost:8000/health

# Access the API at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (if not already trained)
python scripts/train.py

# Run the API server
uvicorn src.decision_tree_predictor.api:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip
- virtualenv (recommended)
- Docker (for containerized deployment)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/prashantmaya/decision-trees-and-hyperparameter-tuning.git
cd decision-tress-and-hyperparameters

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Copy environment variables template
cp .env.example .env

# Ensure data is in place
# The weatherAUS.csv should be in the data/ directory

# Train the model
make train
# or
python scripts/train.py
```

## 💻 Usage

### Starting the API Server

```bash
# Development mode with auto-reload
make run
# or
uvicorn src.decision_tree_predictor.api:app --reload

# Production mode
uvicorn src.decision_tree_predictor.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Making Predictions

#### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```

#### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/predict"
payload = {
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

response = requests.post(url, json=payload)
print(response.json())
# Output: {"will_rain_tomorrow": false, "probability": 0.85, "timestamp": "2025-10-19T..."}
```

## 📚 API Documentation

### Endpoints

| Method | Endpoint                             | Description              |
| ------ | ------------------------------------ | ------------------------ |
| GET    | `/`                                | Root endpoint            |
| GET    | `/health`                          | Health check             |
| POST   | `/api/v1/predict`                  | Make prediction          |
| GET    | `/api/v1/model/info`               | Get model information    |
| GET    | `/api/v1/model/feature-importance` | Get feature importance   |
| GET    | `/docs`                            | Swagger UI documentation |
| GET    | `/redoc`                           | ReDoc documentation      |

### Response Format

**Prediction Response:**

```json
{
  "will_rain_tomorrow": true,
  "probability": 0.85,
  "timestamp": "2025-10-19T12:00:00Z"
}
```

**Model Info Response:**

```json
{
  "model_type": "Decision Tree Classifier",
  "hyperparameters": {
    "max_depth": 7,
    "max_leaf_nodes": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
  },
  "metrics": {...}
}
```

**Feature Importance Response:**

```json
{
  "total_features": 120,
  "top_features": [
    {"feature": "Humidity3pm", "importance": 0.35},
    {"feature": "Pressure3pm", "importance": 0.25},
    ...
  ]
}
```

## 🎓 Model Training

### Training the Model

```bash
# Using make
make train

# Direct execution
python scripts/train.py

# With custom data path (edit config or .env first)
python scripts/train.py
```

### Training Output

The training script will:

1. Load and preprocess the data
2. Split into train/validation sets
3. Train the Decision Tree with specified hyperparameters
4. Calculate and display metrics
5. Save the model and all preprocessing components
6. Export feature importance

### Model Artifacts

After training, the following files are created:

- `model/model.joblib` - Trained Decision Tree model
- `model/imputer.joblib` - Numerical feature imputer
- `model/scaler.joblib` - Feature scaler
- `model/encoder.joblib` - Categorical encoder
- `model/metadata.joblib` - Model metadata and metrics
- `data/feature_importance.csv` - Feature importance scores

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test
# or
pytest

# Run with verbose output
make test-verbose
# or
pytest -v

# Run with coverage report
make test-coverage
# or
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_api.py
pytest tests/test_decision_tree_predictor.py
```

### Test Coverage

View the HTML coverage report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🐳 Docker Deployment

### Building the Image

```bash
# Using Docker Compose
make docker-build
# or
docker-compose build

# Using Docker directly
docker build -t decision-tree-weather-api:latest .
```

### Running the Container

```bash
# Using Docker Compose (recommended)
make docker-run
# or
docker-compose up -d

# View logs
make docker-logs
# or
docker-compose logs -f

# Stop the container
make docker-stop
# or
docker-compose down
```

### Accessing the Containerized API

```bash
# Check health
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d @sample_input.json

# Open browser for interactive docs
open http://localhost:8000/docs
```

## 📁 Project Structure

```
decision-tress-and-hyperparameters/
├── src/
│   └── decision_tree_predictor/
│       ├── __init__.py          # Package initialization
│       ├── api.py               # FastAPI application
│       ├── config.py            # Configuration management
│       ├── model.py             # Decision Tree model class
│       └── utils.py             # Utility functions
├── scripts/
│   └── train.py                 # Model training script
├── tests/
│   ├── __init__.py
│   ├── test_api.py              # API tests
│   └── test_decision_tree_predictor.py  # Model tests
├── data/
│   ├── weatherAUS.csv           # Training data
│   └── feature_importance.csv   # Feature importance (generated)
├── dataset/                     # Original dataset folder
│   ├── weatherAUS.csv
│   ├── x_training.csv
│   ├── x_validation.csv
│   └── ...
├── model/                       # Trained model artifacts
│   ├── model.joblib
│   ├── imputer.joblib
│   ├── scaler.joblib
│   ├── encoder.joblib
│   └── metadata.joblib
├── logs/                        # Application logs
│   └── app.log
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile                   # Docker image definition
├── Makefile                     # Automation commands
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── run_locally.sh               # Local setup script
└── README.md                    # This file
```

## 🔧 Configuration

Configuration is managed through environment variables. Copy `.env.example` to `.env` and customize:

```bash
# Application
APP_NAME=Decision Tree Weather Prediction API
DEBUG=False
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Hyperparameters
MAX_DEPTH=7
MAX_LEAF_NODES=8
MIN_SAMPLES_SPLIT=2
MIN_SAMPLES_LEAF=1
RANDOM_STATE=42

# Paths
MODEL_DIR=model
DATA_DIR=data
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Format code (`make format`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting (120 char line length)
- Use isort for import sorting
- Add type hints where applicable
- Write docstrings for functions and classes
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Weather data from [Australian Government Bureau of Meteorology](http://www.bom.gov.au/)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML powered by [scikit-learn](https://scikit-learn.org/)
- Inspired by production ML best practices

## 📞 Contact

Prashant Soni - [@prashantmaya](https://github.com/prashantmaya)

Project Link: [https://github.com/prashantmaya/decision-trees-and-hyperparameter-tuning](https://github.com/prashantmaya/decision-trees-and-hyperparameter-tuning)

## 📈 Future Enhancements

- [ ] Add more ML models (Random Forest, XGBoost, Gradient Boosting)
- [ ] Implement automated hyperparameter tuning pipeline
- [ ] Add model versioning and A/B testing
- [ ] Implement monitoring and observability (Prometheus, Grafana)
- [ ] Create web UI for predictions
- [ ] Add batch prediction endpoint
- [ ] Implement model retraining pipeline with MLOps
- [ ] Add feature importance visualization dashboard
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add model explainability (SHAP values)
- [ ] Implement caching for faster predictions
- [ ] Add rate limiting and authentication

---

Made with ❤️ using Decision Trees and Python
