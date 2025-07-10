#!/usr/bin/env python3
"""
Database Setup Script for Module 1: Data Foundation
Automated database initialization for student environments
"""
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import config
from src.spatial_database import PostgreSQLSpatialManager
from src.wamex_processor import WAMEXDataProcessor
from src.health_monitor import HealthMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """
    Complete database setup for Module 1
    Measurable Success: Database ready for 1000+ records with <500ms queries
    """
    logger.info("=== Module 1: Database Setup Started ===")
    
    try:
        # Initialize database manager
        db_manager = PostgreSQLSpatialManager()
        
        # Step 1: Setup PostGIS extensions
        logger.info("Step 1: Setting up PostGIS extensions...")
        if db_manager.setup_spatial_extensions():
            logger.info("✅ PostGIS extensions configured successfully")
        else:
            logger.error("❌ Failed to setup PostGIS extensions")
            return False
        
        # Step 2: Create WAMEX schema
        logger.info("Step 2: Creating WAMEX database schema...")
        if db_manager.create_wamex_schema():
            logger.info("✅ WAMEX schema created successfully")
        else:
            logger.error("❌ Failed to create WAMEX schema")
            return False
        
        # Step 3: Create spatial indexes
        logger.info("Step 3: Creating spatial indexes for performance...")
        index_report = db_manager.create_spatial_indexes()
        logger.info(f"✅ Created {index_report.indexes_created} spatial indexes in {index_report.creation_time_seconds:.2f}s")
        
        # Step 4: Validate configuration
        logger.info("Step 4: Validating configuration...")
        validation_report = config.validate_configuration()
        
        if validation_report['database_connection'] and validation_report['postgis_available']:
            logger.info("✅ Configuration validation passed")
            logger.info(f"   PostGIS Version: {validation_report.get('postgis_version', 'Unknown')}")
        else:
            logger.error("❌ Configuration validation failed")
            return False
        
        # Step 5: Health check
        logger.info("Step 5: Running comprehensive health check...")
        health_monitor = HealthMonitor(db_manager)
        health_report = health_monitor.get_comprehensive_health_report()
        
        overall_score = health_report['overall_health_score']
        logger.info(f"✅ Database setup completed - Health Score: {overall_score:.1f}/100")
        
        # Success criteria for supervision
        success_criteria = {
            'database_connection': validation_report['database_connection'],
            'postgis_available': validation_report['postgis_available'],
            'schema_created': True,
            'indexes_created': index_report.indexes_created >= 4,
            'health_score': overall_score >= 80.0
        }
        
        logger.info("=== Setup Success Criteria ===")
        for criterion, met in success_criteria.items():
            status = "✅" if met else "❌"
            logger.info(f"{status} {criterion}: {met}")
        
        all_criteria_met = all(success_criteria.values())
        
        if all_criteria_met:
            logger.info("🎉 Module 1 database setup completed successfully!")
            logger.info("Database is ready for data processing and Module 2 integration")
            return True
        else:
            logger.error("❌ Some setup criteria not met - check configuration")
            return False
        
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        return False

if __name__ == '__main__':
    success = setup_database()
    sys.exit(0 if success else 1)
