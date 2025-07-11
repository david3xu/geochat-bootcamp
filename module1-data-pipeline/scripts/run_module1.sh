#!/bin/bash
# Module 1: Data Foundation - Automated Execution Script

# Exit on error
set -e

echo "===== Module 1: Data Foundation - Execution Script ====="
echo "Starting execution at $(date)"

# Navigate to module directory
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or not accessible"
  exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/make_scripts_executable.sh
./scripts/make_scripts_executable.sh

# Start PostgreSQL + PostGIS database
echo "Starting PostgreSQL + PostGIS database..."
docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
docker-compose -f docker-compose.dev.yml up -d postgis-db

# Wait for database to initialize
echo "Waiting for database to initialize..."
sleep 15

# Setup database schema
echo "Setting up database schema..."
python scripts/setup_database.py

# Load sample data
echo "Loading sample data..."
python scripts/load_sample_data.py

# Run tests
echo "Running tests..."
pytest tests/ -v

# Start API service in background
echo "Starting API service..."
export FLASK_APP=src/data_api.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to start..."
sleep 5

# Test API endpoints
echo "Testing API endpoints..."
echo "Health check:"
curl -s http://localhost:5000/api/health | head -n 20
echo -e "\n\nRecords endpoint:"
curl -s http://localhost:5000/api/data/records?limit=3 | head -n 20
echo -e "\n\nSpatial search endpoint:"
curl -s http://localhost:5000/api/data/spatial-search?lat=-31.9505\&lng=115.8605\&radius=50000 | head -n 20
echo -e "\n\nMinerals endpoint:"
curl -s http://localhost:5000/api/data/minerals | head -n 20

# Performance test
echo -e "\n\nPerformance test:"
echo "Spatial query response time:"
curl -w "%{time_total}s\n" -s http://localhost:5000/api/data/spatial-search?lat=-31.9505\&lng=115.8605\&radius=50000 -o /dev/null

# Stop API service
echo "Stopping API service..."
kill $API_PID

echo "===== Module 1 execution completed successfully! ====="
echo "To start the services manually:"
echo "1. Start database: docker-compose -f docker-compose.dev.yml up -d postgis-db"
echo "2. Start API: flask run --host=0.0.0.0 --port=5000"
echo "3. Access API at: http://localhost:5000/api/health" 