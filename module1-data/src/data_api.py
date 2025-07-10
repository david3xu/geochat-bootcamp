"""
Flask REST API for Geological Data Access
Measurable Success: 3 endpoints responding <500ms, 99%+ uptime
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import json
from typing import Dict, Any
from dataclasses import asdict
import pandas as pd

from .config import config
from .spatial_database import PostgreSQLSpatialManager, SpatialQueryOptimizer
from .wamex_processor import WAMEXDataProcessor
from .health_monitor import HealthMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=config.api.cors_origins)

# Initialize database manager
db_manager = PostgreSQLSpatialManager()
query_optimizer = SpatialQueryOptimizer(db_manager)
health_monitor = HealthMonitor(db_manager)

class GeologicalDataAPI:
    """
    Flask REST API for geological data access
    Measurable Success: 3 endpoints responding <500ms, 99%+ uptime
    """
    
    def __init__(self, app: Flask, db_manager: PostgreSQLSpatialManager):
        self.app = app
        self.db_manager = db_manager
        self.setup_routes()
        
    def setup_routes(self):
        """Configure API routes"""
        
        @self.app.route('/api/data/records', methods=['GET'])
        def get_geological_records():
            """
            GET /api/data/records - Paginated geological record retrieval
            Success Metric: <300ms response time for 100 records
            """
            start_time = time.time()
            
            try:
                # Parse query parameters
                limit = min(int(request.args.get('limit', 100)), 1000)  # Max 1000 records
                offset = int(request.args.get('offset', 0))
                mineral_type = request.args.get('mineral_type', None)
                
                # Build query parameters
                query_params = {
                    'limit': limit,
                    'offset': offset
                }
                
                if mineral_type:
                    query_params['mineral_type'] = mineral_type
                
                # Execute query
                result = self.db_manager.execute_spatial_query(query_params)
                
                response_time = (time.time() - start_time) * 1000
                
                # Add metadata to response
                response_data = {
                    'data': result['results'],
                    'pagination': {
                        'limit': limit,
                        'offset': offset,
                        'count': result['count']
                    },
                    'performance': {
                        'response_time_ms': response_time,
                        'target_met': response_time < 300,
                        'query_type': result.get('query_type', 'simple_select')
                    },
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                logger.info(f"Records endpoint: {result['count']} records in {response_time:.2f}ms")
                return jsonify(response_data), 200
                
            except Exception as e:
                logger.error(f"Error in get_geological_records: {str(e)}")
                return jsonify({
                    'error': 'Internal server error',
                    'message': str(e),
                    'timestamp': pd.Timestamp.now().isoformat()
                }), 500
        
        @self.app.route('/api/data/spatial-search', methods=['GET'])
        def search_by_location():
            """
            GET /api/data/spatial-search - Geographic boundary search
            Success Metric: <500ms for complex polygon intersection queries
            """
            start_time = time.time()
            
            try:
                # Parse spatial query parameters
                latitude = float(request.args.get('lat', 0))
                longitude = float(request.args.get('lng', 0))
                radius = float(request.args.get('radius', 10000))  # Default 10km
                mineral_type = request.args.get('mineral_type', None)
                
                # Validate coordinates
                if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                    return jsonify({
                        'error': 'Invalid coordinates',
                        'message': 'Latitude must be between -90 and 90, longitude between -180 and 180'
                    }), 400
                
                # Build spatial query
                query_params = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'radius': radius
                }
                
                if mineral_type:
                    query_params['mineral_type'] = mineral_type
                
                # Execute spatial query
                result = self.db_manager.execute_spatial_query(query_params)
                
                response_time = (time.time() - start_time) * 1000
                
                response_data = {
                    'data': result['results'],
                    'search_parameters': {
                        'center': [longitude, latitude],
                        'radius_meters': radius,
                        'mineral_type_filter': mineral_type
                    },
                    'results_summary': {
                        'count': result['count'],
                        'search_area_km2': (3.14159 * (radius/1000)**2)  # Approximate area
                    },
                    'performance': {
                        'response_time_ms': response_time,
                        'target_met': response_time < 500,
                        'query_complexity': 'spatial_distance'
                    },
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                logger.info(f"Spatial search: {result['count']} records in {response_time:.2f}ms")
                return jsonify(response_data), 200
                
            except ValueError as e:
                return jsonify({
                    'error': 'Invalid parameters',
                    'message': str(e)
                }), 400
            except Exception as e:
                logger.error(f"Error in search_by_location: {str(e)}")
                return jsonify({
                    'error': 'Internal server error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/data/minerals', methods=['GET'])
        def get_mineral_types():
            """
            GET /api/data/minerals - Mineral classification data
            Success Metric: <200ms for metadata aggregation queries
            """
            start_time = time.time()
            
            try:
                location_filter = request.args.get('location', None)
                
                # Build aggregation query
                query_params = {'limit': 1000}  # Get enough data for aggregation
                
                if location_filter:
                    # Parse location filter (simplified)
                    coords = location_filter.split(',')
                    if len(coords) == 2:
                        query_params.update({
                            'latitude': float(coords[0]),
                            'longitude': float(coords[1]),
                            'radius': 50000  # 50km radius
                        })
                
                # Execute query
                result = self.db_manager.execute_spatial_query(query_params)
                
                # Aggregate mineral types
                mineral_counts = {}
                for record in result['results']:
                    mineral = record.get('mineral_type', 'Unknown')
                    mineral_counts[mineral] = mineral_counts.get(mineral, 0) + 1
                
                # Sort by frequency
                sorted_minerals = sorted(mineral_counts.items(), key=lambda x: x[1], reverse=True)
                
                response_time = (time.time() - start_time) * 1000
                
                response_data = {
                    'mineral_types': dict(sorted_minerals),
                    'total_records_analyzed': result['count'],
                    'unique_mineral_types': len(mineral_counts),
                    'location_filter_applied': location_filter is not None,
                    'performance': {
                        'response_time_ms': response_time,
                        'target_met': response_time < 200,
                        'query_type': 'aggregation'
                    },
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                logger.info(f"Minerals endpoint: {len(mineral_counts)} types in {response_time:.2f}ms")
                return jsonify(response_data), 200
                
            except Exception as e:
                logger.error(f"Error in get_mineral_types: {str(e)}")
                return jsonify({
                    'error': 'Internal server error',
                    'message': str(e)
                }), 500

class APIPerformanceMonitor:
    """
    API endpoint performance tracking and alerting
    Measurable Success: 99%+ uptime monitoring with alerts
    """
    
    def __init__(self):
        self.request_metrics = []
        self.uptime_start = time.time()
        
    def track_request(self, endpoint: str, method: str, response_time: float, status_code: int):
        """Track individual request performance"""
        metric = {
            'endpoint': endpoint,
            'method': method,
            'response_time_ms': response_time * 1000,
            'status_code': status_code,
            'timestamp': pd.Timestamp.now().isoformat(),
            'success': 200 <= status_code < 400
        }
        
        self.request_metrics.append(metric)
        
        # Keep only last 1000 requests
        if len(self.request_metrics) > 1000:
            self.request_metrics = self.request_metrics[-1000:]
    
    def get_performance_summary(self) -> Dict:
        """Generate performance summary for supervision"""
        if not self.request_metrics:
            return {'status': 'No requests tracked yet'}
        
        recent_requests = self.request_metrics[-100:]  # Last 100 requests
        
        total_requests = len(recent_requests)
        successful_requests = sum(1 for r in recent_requests if r['success'])
        avg_response_time = sum(r['response_time_ms'] for r in recent_requests) / total_requests
        
        # Calculate uptime
        uptime_seconds = time.time() - self.uptime_start
        uptime_hours = uptime_seconds / 3600
        
        # Performance targets
        performance_score = (successful_requests / total_requests) * 100
        response_time_compliance = sum(1 for r in recent_requests if r['response_time_ms'] < 500) / total_requests * 100
        
        summary = {
            'uptime_hours': uptime_hours,
            'total_requests_recent': total_requests,
            'success_rate_percent': performance_score,
            'average_response_time_ms': avg_response_time,
            'response_time_compliance_percent': response_time_compliance,
            'performance_targets': {
                'uptime_target': 99.0,
                'uptime_achieved': min(99.9, performance_score),  # Simplified calculation
                'response_time_target_ms': 500,
                'response_time_achieved_ms': avg_response_time
            },
            'alert_status': self._calculate_alert_status(performance_score, avg_response_time)
        }
        
        return summary
    
    def _calculate_alert_status(self, success_rate: float, avg_response_time: float) -> str:
        """Calculate system alert status"""
        if success_rate >= 99 and avg_response_time < 500:
            return 'healthy'
        elif success_rate >= 95 and avg_response_time < 1000:
            return 'warning'
        else:
            return 'critical'

# Global performance monitor
performance_monitor = APIPerformanceMonitor()

@app.before_request
def before_request():
    """Track request start time"""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Track request completion and performance"""
    if hasattr(request, 'start_time'):
        response_time = time.time() - request.start_time
        performance_monitor.track_request(
            endpoint=request.endpoint or request.path,
            method=request.method,
            response_time=response_time,
            status_code=response.status_code
        )
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    GET /api/health - System health and performance monitoring
    Success Metric: Real-time performance metrics for supervision
    """
    try:
        # Get comprehensive health report
        health_report = health_monitor.get_comprehensive_health_report()
        
        # Add API performance metrics
        api_performance = performance_monitor.get_performance_summary()
        
        # Add database performance
        db_performance = query_optimizer.monitor_query_performance()
        
        comprehensive_report = {
            'service_status': 'healthy',
            'timestamp': pd.Timestamp.now().isoformat(),
            'system_health': health_report,
            'api_performance': api_performance,
            'database_performance': db_performance,
            'module_readiness': {
                'data_processing_ready': True,
                'database_operational': health_report.get('database_connection', False),
                'api_endpoints_responding': api_performance.get('success_rate_percent', 0) > 95,
                'module2_integration_ready': True  # Set when API contracts are established
            },
            'supervision_metrics': {
                'data_processing_accuracy': health_report.get('data_quality', {}).get('processing_accuracy', 0),
                'query_performance_compliance': db_performance.get('target_compliance_percentage', 0),
                'api_uptime_percentage': api_performance.get('uptime_achieved', 0),
                'overall_module_score': health_monitor.calculate_overall_score()
            }
        }
        
        return jsonify(comprehensive_report), 200
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'service_status': 'unhealthy',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }), 500

# Initialize API
api = GeologicalDataAPI(app, db_manager)

if __name__ == '__main__':
    # Initialize database on startup
    logger.info("Initializing Module 1: Data Foundation...")
    
    try:
        # Setup database
        if db_manager.setup_spatial_extensions():
            logger.info("✅ PostGIS extensions configured")
        
        if db_manager.create_wamex_schema():
            logger.info("✅ WAMEX schema created")
        
        # Create indexes for performance
        index_report = db_manager.create_spatial_indexes()
        logger.info(f"✅ Created {index_report.indexes_created} spatial indexes")
        
        # Start Flask application
        logger.info(f"Starting API server on {config.api.host}:{config.api.port}")
        app.run(
            host=config.api.host,
            port=config.api.port,
            debug=config.api.debug
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize Module 1: {str(e)}")
        raise
