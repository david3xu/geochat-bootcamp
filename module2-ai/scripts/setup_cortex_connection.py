#!/usr/bin/env python3
"""
Snowflake Cortex Connection Setup Script
Module 2: AI Engine - Week 2 Implementation
"""
import yaml
import os
import sys
from pathlib import Path

def setup_cortex_connection():
    """Setup Snowflake Cortex connection for Module 2"""
    config_path = Path(__file__).parent.parent / "config" / "snowflake_credentials.yml"
    
    if not config_path.exists():
        print("Error: snowflake_credentials.yml not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        required_fields = ['account', 'user', 'password', 'warehouse', 'database', 'schema']
        snowflake_config = config.get('snowflake', {})
        
        for field in required_fields:
            if field not in snowflake_config:
                print(f"Error: Missing required field '{field}' in configuration")
                return False
        
        print("Snowflake Cortex connection setup completed successfully")
        return True
        
    except Exception as e:
        print(f"Error setting up Cortex connection: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_cortex_connection()
    sys.exit(0 if success else 1)
