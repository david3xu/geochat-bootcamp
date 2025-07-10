# GeoChat Bootcamp - Documentation

This directory contains documentation for the GeoChat Bootcamp project.

## Directory Structure

- **execution_guides/**: Step-by-step guides for running each module
  - [Module 1: Data Foundation](execution_guides/module1_execution_guide.md)

## Project Overview

GeoChat is a geospatial analytics platform for the mining industry that provides:
- Geological data processing and analysis
- Spatial database management
- REST API for data access
- Monitoring and health checks

## Modules

1. **Module 1: Data Foundation**
   - Processes geological WAMEX data with 98%+ accuracy
   - Implements a spatial database with PostGIS
   - Provides a Flask REST API with three endpoints
   - Includes health monitoring and performance metrics

## Getting Started

To get started with the GeoChat Bootcamp project:

1. Review the [Module 1 Execution Guide](execution_guides/module1_execution_guide.md)
2. Navigate to the module1-data directory
3. Run the automated setup script: `./scripts/run_module1.sh`

## Related Resources

- Module scripts are located in each module's `scripts` directory
- Configuration files are in each module's `config` directory
- Sample data is provided in each module's `data` directory 