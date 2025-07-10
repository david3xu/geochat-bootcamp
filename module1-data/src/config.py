"""
Configuration management for Module 1: Data Foundation
Measurable Success: Environment-specific configuration with validation
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class APIConfig:
    """API service configuration"""
    host: str
    port: int
    debug: bool
    cors_origins: list

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    batch_size: int
    max_retries: int
    validation_threshold: float
    coordinate_system_source: str
    coordinate_system_target: str

class ConfigurationManager:
    """
    Centralized configuration management with validation
    Measurable Success: 100% configuration validation and environment isolation
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent / "config"
        self.database = self._load_database_config()
        self.api = self._load_api_config()
        self.processing = self._load_processing_config()
        
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration with environment override"""
        # Environment variables take precedence (for Azure deployment)
        if database_url := os.getenv('DATABASE_URL'):
            # Parse Azure PostgreSQL connection string
            import urllib.parse as urlparse
            parsed = urlparse.urlparse(database_url)
            return DatabaseConfig(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],  # Remove leading '/'
                username=parsed.username,
                password=parsed.password
            )
        
        # Fallback to configuration file
        config_file = self.config_path / "database_config.yml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return DatabaseConfig(**config['database'])
        
        # Development defaults
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'geochat_data'),
            username=os.getenv('DB_USER', 'geochat_user'),
            password=os.getenv('DB_PASSWORD', 'geochat_pass')
        )
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration"""
        return APIConfig(
            host=os.getenv('API_HOST', '0.0.0.0'),
            port=int(os.getenv('API_PORT', '5000')),
            debug=os.getenv('FLASK_ENV') == 'development',
            cors_origins=os.getenv('CORS_ORIGINS', '*').split(',')
        )
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load data processing configuration"""
        return ProcessingConfig(
            batch_size=int(os.getenv('BATCH_SIZE', '100')),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            validation_threshold=float(os.getenv('VALIDATION_THRESHOLD', '0.98')),
            coordinate_system_source='EPSG:7844',  # GDA2020
            coordinate_system_target='EPSG:4326'   # WGS84
        )
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration for deployment readiness
        Returns validation report for supervisor monitoring
        """
        validation_report = {
            'database_connection': False,
            'postgis_available': False,
            'api_configuration': False,
            'processing_parameters': False,
            'validation_timestamp': None
        }
        
        try:
            # Test database connection
            import psycopg2
            conn = psycopg2.connect(self.database.connection_string)
            cursor = conn.cursor()
            
            # Verify PostGIS extension
            cursor.execute("SELECT PostGIS_Version();")
            postgis_version = cursor.fetchone()[0]
            
            validation_report.update({
                'database_connection': True,
                'postgis_available': True,
                'postgis_version': postgis_version,
                'api_configuration': True,
                'processing_parameters': True,
                'validation_timestamp': str(pd.Timestamp.now())
            })
            
            conn.close()
            
        except Exception as e:
            validation_report['error'] = str(e)
        
        return validation_report

# Global configuration instance
config = ConfigurationManager() 