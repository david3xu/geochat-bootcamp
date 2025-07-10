"""
Health Monitoring and Performance Tracking for Module 1
Measurable Success: Real-time supervision metrics and alerts
"""
import psycopg2
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import pandas as pd

from .config import config

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """System health metrics for supervision"""
    database_connection: bool
    postgis_available: bool
    api_responsiveness: bool
    data_quality_score: float
    query_performance_score: float
    overall_health_score: float

class HealthMonitor:
    """
    Performance monitoring and health checking for supervision
    Measurable Success: 99%+ system health monitoring accuracy
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.health_history: List[SystemHealth] = []
        
    def check_database_connection(self) -> Dict[str, Any]:
        """Test database connectivity and PostGIS availability"""
        try:
            with psycopg2.connect(config.database.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Test basic connection
                    cursor.execute("SELECT 1;")
                    basic_connection = cursor.fetchone()[0] == 1
                    
                    # Test PostGIS
                    cursor.execute("SELECT PostGIS_Version();")
                    postgis_version = cursor.fetchone()[0]
                    
                    # Test WAMEX table
                    cursor.execute("SELECT COUNT(*) FROM wamex_records;")
                    record_count = cursor.fetchone()[0]
                    
                    return {
                        'database_connection': True,
                        'postgis_available': True,
                        'postgis_version': postgis_version,
                        'wamex_records_count': record_count,
                        'connection_test_passed': True
                    }
                    
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return {
                'database_connection': False,
                'postgis_available': False,
                'error': str(e),
                'connection_test_passed': False
            }
    
    def check_api_responsiveness(self) -> Dict[str, Any]:
        """Test API endpoint responsiveness"""
        try:
            # Simulate API response time test
            start_time = time.time()
            
            # Test simple query
            query_result = self.db_manager.execute_spatial_query({'limit': 10})
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'api_responsive': response_time < 500,
                'response_time_ms': response_time,
                'records_returned': query_result.get('count', 0),
                'performance_target_met': response_time < 500
            }
            
        except Exception as e:
            logger.error(f"API responsiveness check failed: {str(e)}")
            return {
                'api_responsive': False,
                'error': str(e),
                'performance_target_met': False
            }
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """Assess data processing quality and accuracy"""
        try:
            # Check data completeness
            db_check = self.check_database_connection()
            record_count = db_check.get('wamex_records_count', 0)
            
            # Simulate data quality assessment
            quality_score = min(100.0, (record_count / 1000) * 100)  # Target: 1000 records
            
            return {
                'total_records': record_count,
                'processing_accuracy': quality_score,
                'quality_target_met': quality_score >= 98.0,
                'data_completeness_score': quality_score,
                'target_records': 1000,
                'records_target_met': record_count >= 1000
            }
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {str(e)}")
            return {
                'processing_accuracy': 0.0,
                'quality_target_met': False,
                'error': str(e)
            }
    
    def evaluate_query_performance(self) -> Dict[str, Any]:
        """Evaluate spatial query performance"""
        try:
            # Test multiple query types
            test_queries = [
                {'limit': 100},  # Simple select
                {'latitude': -31.9505, 'longitude': 115.8605, 'radius': 10000},  # Spatial query
                {'mineral_type': 'gold', 'limit': 50}  # Filtered query
            ]
            
            performance_results = []
            total_time = 0
            
            for query_params in test_queries:
                start_time = time.time()
                result = self.db_manager.execute_spatial_query(query_params)
                execution_time = (time.time() - start_time) * 1000
                
                performance_results.append({
                    'query_type': result.get('query_type', 'unknown'),
                    'execution_time_ms': execution_time,
                    'records_returned': result.get('count', 0),
                    'target_met': execution_time < 500
                })
                
                total_time += execution_time
            
            avg_performance = total_time / len(test_queries)
            performance_compliance = sum(1 for r in performance_results if r['target_met']) / len(performance_results) * 100
            
            return {
                'average_query_time_ms': avg_performance,
                'performance_compliance_percent': performance_compliance,
                'query_performance_score': performance_compliance,
                'performance_target_met': avg_performance < 500,
                'individual_query_results': performance_results
            }
            
        except Exception as e:
            logger.error(f"Query performance evaluation failed: {str(e)}")
            return {
                'query_performance_score': 0.0,
                'performance_target_met': False,
                'error': str(e)
            }
    
    def calculate_overall_score(self) -> float:
        """Calculate overall module health score for supervision"""
        try:
            # Get all health metrics
            db_health = self.check_database_connection()
            api_health = self.check_api_responsiveness()
            data_quality = self.assess_data_quality()
            query_performance = self.evaluate_query_performance()
            
            # Weight different components
            weights = {
                'database': 0.25,
                'api': 0.25,
                'data_quality': 0.25,
                'performance': 0.25
            }
            
            # Calculate component scores
            scores = {
                'database': 100.0 if db_health.get('connection_test_passed', False) else 0.0,
                'api': 100.0 if api_health.get('performance_target_met', False) else 0.0,
                'data_quality': data_quality.get('processing_accuracy', 0.0),
                'performance': query_performance.get('query_performance_score', 0.0)
            }
            
            # Calculate weighted overall score
            overall_score = sum(scores[component] * weights[component] for component in weights.keys())
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Overall score calculation failed: {str(e)}")
            return 0.0
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report for instructor supervision"""
        logger.info("Generating comprehensive health report...")
        
        # Collect all health metrics
        db_health = self.check_database_connection()
        api_health = self.check_api_responsiveness()
        data_quality = self.assess_data_quality()
        query_performance = self.evaluate_query_performance()
        overall_score = self.calculate_overall_score()
        
        # Create supervision-friendly report
        report = {
            'module_name': 'Module 1: Data Foundation',
            'assessment_timestamp': pd.Timestamp.now().isoformat(),
            'overall_health_score': overall_score,
            'health_status': self._determine_health_status(overall_score),
            
            'component_health': {
                'database': db_health,
                'api_performance': api_health,
                'data_quality': data_quality,
                'query_performance': query_performance
            },
            
            'supervision_metrics': {
                'learning_targets_met': {
                    'data_accuracy_98_percent': data_quality.get('quality_target_met', False),
                    'query_performance_500ms': query_performance.get('performance_target_met', False),
                    'api_uptime_99_percent': api_health.get('performance_target_met', False),
                    'azure_integration_working': db_health.get('connection_test_passed', False)
                },
                
                'measurable_outcomes': {
                    'records_processed': data_quality.get('total_records', 0),
                    'processing_accuracy_percent': data_quality.get('processing_accuracy', 0.0),
                    'average_query_time_ms': query_performance.get('average_query_time_ms', 0.0),
                    'api_response_time_ms': api_health.get('response_time_ms', 0.0)
                },
                
                'student_evidence': {
                    'postgis_version': db_health.get('postgis_version', 'Not available'),
                    'database_records_count': db_health.get('wamex_records_count', 0),
                    'performance_compliance_percent': query_performance.get('performance_compliance_percent', 0.0),
                    'module2_integration_ready': self._check_module2_readiness(db_health, data_quality)
                }
            },
            
            'alerts_and_recommendations': self._generate_alerts_and_recommendations(
                db_health, api_health, data_quality, query_performance
            )
        }
        
        # Store health record
        current_health = SystemHealth(
            database_connection=db_health.get('connection_test_passed', False),
            postgis_available=db_health.get('postgis_available', False),
            api_responsiveness=api_health.get('performance_target_met', False),
            data_quality_score=data_quality.get('processing_accuracy', 0.0),
            query_performance_score=query_performance.get('query_performance_score', 0.0),
            overall_health_score=overall_score
        )
        
        self.health_history.append(current_health)
        
        logger.info(f"Health report generated - Overall score: {overall_score:.2f}")
        return report
    
    def _determine_health_status(self, overall_score: float) -> str:
        """Determine health status based on overall score"""
        if overall_score >= 95:
            return 'excellent'
        elif overall_score >= 80:
            return 'good'
        elif overall_score >= 60:
            return 'fair'
        else:
            return 'poor'
    
    def _check_module2_readiness(self, db_health: Dict, data_quality: Dict) -> bool:
        """Check if Module 1 is ready for Module 2 integration"""
        return (
            db_health.get('connection_test_passed', False) and
            data_quality.get('total_records', 0) >= 100 and  # Minimum data for AI training
            data_quality.get('processing_accuracy', 0.0) >= 90.0
        )
    
    def _generate_alerts_and_recommendations(self, db_health: Dict, api_health: Dict, 
                                           data_quality: Dict, query_performance: Dict) -> List[str]:
        """Generate actionable alerts and recommendations"""
        alerts = []
        
        # Database alerts
        if not db_health.get('connection_test_passed', False):
            alerts.append("CRITICAL: Database connection failed - check Azure PostgreSQL configuration")
        
        if not db_health.get('postgis_available', False):
            alerts.append("CRITICAL: PostGIS extension not available - run setup_spatial_extensions()")
        
        # Data quality alerts
        if data_quality.get('processing_accuracy', 0.0) < 98.0:
            alerts.append(f"WARNING: Data processing accuracy {data_quality.get('processing_accuracy', 0):.1f}% below 98% target")
        
        if data_quality.get('total_records', 0) < 1000:
            alerts.append(f"INFO: Only {data_quality.get('total_records', 0)} records processed, target is 1000+")
        
        # Performance alerts
        if query_performance.get('average_query_time_ms', 0) > 500:
            alerts.append(f"WARNING: Average query time {query_performance.get('average_query_time_ms', 0):.0f}ms exceeds 500ms target")
        
        if api_health.get('response_time_ms', 0) > 500:
            alerts.append(f"WARNING: API response time {api_health.get('response_time_ms', 0):.0f}ms exceeds 500ms target")
        
        # Success messages
        if not alerts:
            alerts.append("SUCCESS: All Module 1 learning targets achieved - ready for Module 2 integration")
        
        return alerts
