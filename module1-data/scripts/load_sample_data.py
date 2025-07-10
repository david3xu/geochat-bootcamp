#!/usr/bin/env python3
"""
Sample Data Loading Script for Module 1
Load and process sample WAMEX data for testing and development
"""
import sys
import os
import logging
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wamex_processor import WAMEXDataProcessor
from src.spatial_database import PostgreSQLSpatialManager
from src.health_monitor import HealthMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data(csv_file: str = None):
    """
    Load sample WAMEX data into database
    Measurable Success: 1,000 records loaded with 98%+ accuracy
    """
    logger.info("=== Module 1: Sample Data Loading Started ===")
    
    try:
        # Initialize processors
        data_processor = WAMEXDataProcessor()
        db_manager = PostgreSQLSpatialManager()
        
        # Use default sample data if not specified
        if not csv_file:
            csv_file = str(Path(__file__).parent.parent / 'data' / 'sample_wamex.csv')
            
        logger.info(f"Using sample data from: {csv_file}")
        
        # Step 1: Load and validate data
        logger.info("Step 1: Loading and validating sample data...")
        raw_data = data_processor.load_csv_data(csv_file)
        logger.info(f"✅ Loaded {len(raw_data)} records from sample data")
        
        # Step 2: Validate spatial coordinates
        logger.info("Step 2: Validating spatial coordinates...")
        validation_report = data_processor.validate_spatial_coordinates(raw_data)
        
        coordinate_accuracy = validation_report.coordinate_accuracy
        logger.info(f"✅ Coordinate validation: {coordinate_accuracy:.1f}% accuracy")
        
        if coordinate_accuracy < 80.0:
            logger.warning(f"⚠️ Low coordinate accuracy ({coordinate_accuracy:.1f}%) - check sample data")
        
        # Step 3: Transform coordinate system
        logger.info("Step 3: Transforming coordinate system...")
        transformed_data = data_processor.transform_coordinate_system(raw_data)
        logger.info("✅ Coordinate transformation completed")
        
        # Step 4: Insert into database
        logger.info("Step 4: Inserting data into database...")
        insert_result = db_manager.insert_geological_records(transformed_data)
        
        insertion_success_rate = insert_result.get('insertion_success_rate', 0.0)
        records_inserted = insert_result.get('records_inserted', 0)
        
        logger.info(f"✅ Database insertion: {records_inserted} records inserted ({insertion_success_rate:.1f}% success)")
        
        # Step 5: Generate metadata report
        logger.info("Step 5: Extracting geological metadata...")
        metadata = data_processor.extract_geological_metadata(transformed_data)
        
        mineral_types = len(metadata.get('mineral_types', {}))
        logger.info(f"✅ Metadata extraction: Found {mineral_types} unique mineral types")
        
        # Step 6: Update processing metrics
        data_processor.metrics.processed_successfully = records_inserted
        data_processor.metrics.failed_records = len(raw_data) - records_inserted
        data_processor.metrics.processing_accuracy = insertion_success_rate
        
        processing_report = data_processor.generate_processing_report()
        
        # Step 7: Health check
        logger.info("Step 7: Verifying data load with health check...")
        health_monitor = HealthMonitor(db_manager)
        health_report = health_monitor.get_comprehensive_health_report()
        
        # Success criteria for supervision
        success_criteria = {
            'records_loaded': records_inserted >= 20,  # Should be 1000 in production
            'insertion_success_rate': insertion_success_rate >= 98.0,
            'coordinate_accuracy': coordinate_accuracy >= 98.0,
            'metadata_extracted': bool(mineral_types > 0),
            'health_score': health_report['overall_health_score'] >= 80.0
        }
        
        logger.info("=== Data Loading Success Criteria ===")
        for criterion, met in success_criteria.items():
            status = "✅" if met else "❌"
            logger.info(f"{status} {criterion}: {met}")
            
        all_criteria_met = all(success_criteria.values())
        
        if all_criteria_met:
            logger.info("🎉 Sample data loaded successfully!")
            logger.info(f"Total records in database: {insert_result.get('total_records_in_database', 0)}")
            return True
        else:
            logger.warning("⚠️ Some loading criteria not met - check sample data and database")
            return False
            
    except Exception as e:
        logger.error(f"Sample data loading failed: {str(e)}")
        return False

if __name__ == '__main__':
    # Accept CSV file path as command line argument
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    success = load_sample_data(csv_file)
    sys.exit(0 if success else 1)
