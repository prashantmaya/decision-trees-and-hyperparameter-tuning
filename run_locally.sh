#!/bin/bash

# run_locally.sh - Setup and run the Decision Tree Weather Prediction API locally

set -e  # Exit on error

echo "=================================================="
echo "Decision Tree Weather Prediction API - Local Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.11"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.11+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
print_info "Installing dependencies..."
pip install -r requirements.txt
print_success "Dependencies installed"

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p logs data model
print_success "Directories created"

# Copy data if needed
if [ ! -f "data/weatherAUS.csv" ] && [ -f "dataset/weatherAUS.csv" ]; then
    print_info "Copying dataset to data directory..."
    cp dataset/weatherAUS.csv data/weatherAUS.csv
    print_success "Dataset copied"
elif [ -f "data/weatherAUS.csv" ]; then
    print_success "Dataset already exists in data directory"
else
    print_error "Dataset not found! Please ensure weatherAUS.csv is in data/ or dataset/ directory"
    exit 1
fi

# Check if .env exists, if not create from example
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    print_info "Creating .env file from template..."
    cp .env.example .env
    print_success ".env file created"
fi

# Check if model exists
if [ ! -f "model/model.joblib" ]; then
    print_info "Trained model not found. Training model..."
    echo ""
    python scripts/train.py
    echo ""
    print_success "Model training completed"
else
    print_success "Trained model found"
fi

# Display instructions
echo ""
echo "=================================================="
echo "Setup completed successfully!"
echo "=================================================="
echo ""
echo "To start the API server, run:"
echo ""
echo "  source venv/bin/activate"
echo "  make run"
echo ""
echo "Or directly:"
echo ""
echo "  source venv/bin/activate"
echo "  uvicorn src.decision_tree_predictor.api:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Then visit:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
echo ""
echo "=================================================="

# Ask if user wants to start the server now
read -p "Do you want to start the API server now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting API server..."
    echo ""
    uvicorn src.decision_tree_predictor.api:app --reload --host 0.0.0.0 --port 8000
fi

