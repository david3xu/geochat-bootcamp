# Module 1: Data Foundation - Execution Guide

This guide provides step-by-step instructions for running Module 1 of the GeoChat Bootcamp.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ installed
- Git repository cloned

## Step 1: Set Up Environment

First, create and activate a Python virtual environment:

```bash
# Navigate to the module1-data directory
cd module1-data

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** The scripts have been updated to work directly with `python scripts/setup_database.py` without needing `PYTHONPATH=.`

## Step 2: Start the PostgreSQL + PostGIS Database

Use Docker Compose to start the database:

```bash
# Start the PostgreSQL + PostGIS container
docker-compose -f docker-compose.dev.yml up -d postgis-db

# Wait for the database to initialize (about 10-15 seconds)
sleep 15
```

## Step 3: Set Up the Database Schema

Run the setup script to create the necessary database schema and extensions:

```bash
# Make sure scripts are executable
chmod +x scripts/make_scripts_executable.sh
./scripts/make_scripts_executable.sh

# Run the database setup script
python scripts/setup_database.py
```

**Expected Output:**
```
INFO:__main__:✅ PostGIS extensions configured successfully
INFO:__main__:✅ WAMEX schema created successfully
INFO:__main__:✅ Created 4 spatial indexes in 0.00s
INFO:__main__:✅ Configuration validation passed
```

## Step 4: Load Sample Data

Load the sample WAMEX geological data into the database:

```bash
# Load the sample data
python scripts/load_sample_data.py
```

**Expected Output:**
```
INFO:__main__:✅ Loaded 20 records from sample data
INFO:__main__:✅ Coordinate validation: 100.0% accuracy
INFO:__main__:✅ Database insertion: 20 records inserted (100.0% success)
INFO:__main__:✅ Metadata extraction: Found 13 unique mineral types
```

## Step 5: Start the API Service

Start the Flask API service:

```bash
# Start the API service
export FLASK_APP=src/data_api.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

The API will be available at http://localhost:5000

## Step 6: Test the API Endpoints

You can test the API endpoints using curl or a web browser:

```bash
# Get health status
curl http://localhost:5000/api/health

# Get geological records
curl http://localhost:5000/api/data/records?limit=10

# Perform a spatial search
curl http://localhost:5000/api/data/spatial-search?lat=-31.9505&lng=115.8605&radius=50000

# Get mineral types
curl http://localhost:5000/api/data/minerals
```

## Step 7: Run Tests

Run the unit tests to verify functionality:

```bash
# Run the tests
pytest tests/
```

## Using Docker Compose for Full Stack

To run the entire stack (database + API) using Docker Compose:

```bash
# Build and start all services
docker-compose -f docker-compose.dev.yml up --build
```

## Automated Execution

For a quick and automated setup, you can use the provided shell script:

```bash
# Navigate to the module1-data directory
cd module1-data

# Make the script executable (if not already)
chmod +x scripts/run_module1.sh

# Run the script
./scripts/run_module1.sh
```

The script will automatically:
1. Set up the required environment
2. Start necessary services
3. Run tests to verify functionality
4. Demonstrate API endpoints
5. Measure performance

## Monitoring and Validation

Check the system health and performance:

```bash
# Get comprehensive health report
curl http://localhost:5000/api/health | python -m json.tool

# Test query performance
curl -w "%{time_total}s\n" -s http://localhost:5000/api/data/spatial-search?lat=-31.9505&lng=115.8605&radius=50000
```

## Expected Performance Metrics

When running successfully, you should see:
- **Database queries**: <10ms average (target: <500ms)
- **API response time**: <5ms (target: <500ms)
- **Health score**: 75-80/100 (limited by sample size)
- **Data accuracy**: 100% (for sample data)

## Stopping the Services

To stop the running services:

```bash
# If running with Flask directly
# Press Ctrl+C to stop the Flask server

# If running with Docker Compose
docker-compose -f docker-compose.dev.yml down
```

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

```bash
# Check if the database container is running
docker ps | grep postgis

# Check database logs
docker logs geochat-postgis

# Restart the database
docker-compose -f docker-compose.dev.yml restart postgis-db
```

### Configuration Validation Issues

If you see "Configuration validation failed":

```bash
# Check if pandas is installed
pip install pandas

# Verify the config files exist
ls -la config/

# Test configuration directly
python -c "from src.config import config; print(config.validate_configuration())"
```

### API Issues

If the API is not responding correctly:

```bash
# Check for Python errors in the console
# Verify database connection settings in config/database_config.yml
# Ensure the database is running and accessible

# Test database connection
python -c "from src.spatial_database import PostgreSQLSpatialManager; db = PostgreSQLSpatialManager(); print('Database connection successful')"
```

### Data Loading Issues

If data loading fails:

```bash
# Check the sample data file
cat data/sample_wamex.csv

# Run the data loader with verbose logging
python scripts/load_sample_data.py
```

### Import Issues

If you see "ModuleNotFoundError: No module named 'src'":

```bash
# The scripts have been updated to handle this automatically
# Just run: python scripts/setup_database.py
# No need for PYTHONPATH=. anymore
```

## Success Criteria

Module 1 is successfully running when:
- ✅ Database setup completes without errors
- ✅ Sample data loads successfully (20+ records)
- ✅ API responds to all endpoints
- ✅ Health score is >70/100
- ✅ Query performance is <500ms
- ✅ All 3 API endpoints return valid JSON responses 