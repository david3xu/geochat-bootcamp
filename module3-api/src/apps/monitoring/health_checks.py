"""
Health check system for monitoring service status
Full Stack AI Engineer Bootcamp - Module 3
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, List
from django.conf import settings
from django.db import connection
from django.core.cache import cache
from channels.layers import get_channel_layer

logger = logging.getLogger('chat2map')

class HealthCheckManager:
    """
    Comprehensive health check manager
    Learning Outcome: System monitoring and observability
    
    Measurable Success Criteria:
    - Health checks: <1 second response time
    - Service availability: 99.9% uptime monitoring
    - Error detection: Real-time failure alerts
    """
    
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'redis': self._check_redis,
            'websocket': self._check_websocket,
            'module1_api': self._check_module1,
            'module2_ai': self._check_module2,
            'external_services': self._check_external_services
        }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks
        Learning Outcome: Comprehensive system health monitoring
        """
        start_time = time.time()
        results = {}
        
        # Run all checks concurrently
        tasks = []
        for check_name, check_func in self.checks.items():
            task = asyncio.create_task(self._run_single_check(check_name, check_func))
            tasks.append(task)
        
        # Wait for all checks to complete
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(check_results):
            check_name = list(self.checks.keys())[i]
            if isinstance(result, Exception):
                results[check_name] = {
                    'status': 'error',
                    'error': str(result),
                    'response_time': 0
                }
            else:
                results[check_name] = result
        
        # Calculate overall health
        overall_status = self._calculate_overall_status(results)
        
        return {
            'overall_status': overall_status,
            'checks': results,
            'total_response_time': time.time() - start_time,
            'timestamp': time.time()
        }
    
    async def _run_single_check(self, check_name: str, check_func) -> Dict[str, Any]:
        """Run a single health check with timing"""
        start_time = time.time()
        
        try:
            result = await check_func()
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy' if result else 'unhealthy',
                'response_time': response_time,
                'details': result if isinstance(result, dict) else {}
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """
        Check database connectivity and performance
        Learning Outcome: Database health monitoring
        """
        try:
            start_time = time.time()
            
            # Test database connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            query_time = time.time() - start_time
            
            # Check PostGIS extension
            with connection.cursor() as cursor:
                cursor.execute("SELECT PostGIS_version()")
                postgis_version = cursor.fetchone()[0]
            
            return {
                'connected': True,
                'query_time': query_time,
                'postgis_version': postgis_version,
                'database_name': settings.DATABASES['default']['NAME']
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {'connected': False, 'error': str(e)}
    
    async def _check_redis(self) -> Dict[str, Any]:
        """
        Check Redis connectivity for WebSocket channels
        Learning Outcome: Cache and messaging system monitoring
        """
        try:
            # Test cache connection
            test_key = 'health_check_test'
            cache.set(test_key, 'test_value', 60)
            value = cache.get(test_key)
            cache.delete(test_key)
            
            # Test channel layer
            channel_layer = get_channel_layer()
            if channel_layer:
                return {
                    'cache_connected': value == 'test_value',
                    'channel_layer_connected': True,
                    'backend': str(type(channel_layer))
                }
            else:
                return {
                    'cache_connected': value == 'test_value',
                    'channel_layer_connected': False
                }
                
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {'connected': False, 'error': str(e)}
    
    async def _check_websocket(self) -> Dict[str, Any]:
        """
        Check WebSocket functionality
        Learning Outcome: Real-time communication monitoring
        """
        try:
            channel_layer = get_channel_layer()
            if not channel_layer:
                return {'available': False, 'error': 'No channel layer configured'}
            
            # Test channel communication
            test_channel = 'health_check_test'
            test_message = {'type': 'health_check', 'data': 'test'}
            
            await channel_layer.send(test_channel, test_message)
            
            return {
                'available': True,
                'channel_layer_type': str(type(channel_layer)),
                'test_successful': True
            }
            
        except Exception as e:
            logger.error(f"WebSocket health check failed: {e}")
            return {'available': False, 'error': str(e)}
    
    async def _check_module1(self) -> Dict[str, Any]:
        """
        Check Module 1 (Data Pipeline) connectivity
        Learning Outcome: Microservice dependency monitoring
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(
                    f"{settings.MODULE1_API_URL}/health"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'connected': True,
                            'status': data.get('status', 'unknown'),
                            'response_time': data.get('response_time', 0)
                        }
                    else:
                        return {
                            'connected': False,
                            'status_code': response.status
                        }
                        
        except Exception as e:
            logger.error(f"Module 1 health check failed: {e}")
            return {'connected': False, 'error': str(e)}
    
    async def _check_module2(self) -> Dict[str, Any]:
        """
        Check Module 2 (AI Intelligence) connectivity
        Learning Outcome: AI service dependency monitoring
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(
                    f"{settings.MODULE2_AI_URL}/health"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'connected': True,
                            'ai_service_status': data.get('ai_status', 'unknown'),
                            'cortex_available': data.get('cortex_available', False),
                            'response_time': data.get('response_time', 0)
                        }
                    else:
                        return {
                            'connected': False,
                            'status_code': response.status
                        }
                        
        except Exception as e:
            logger.error(f"Module 2 health check failed: {e}")
            return {'connected': False, 'error': str(e)}
    
    async def _check_external_services(self) -> Dict[str, Any]:
        """
        Check external service dependencies
        Learning Outcome: External dependency monitoring
        """
        try:
            services = {}
            
            # Check Snowflake Cortex if configured
            if settings.SNOWFLAKE_CORTEX_URL:
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as session:
                        async with session.get(
                            f"{settings.SNOWFLAKE_CORTEX_URL}/health"
                        ) as response:
                            services['snowflake_cortex'] = {
                                'connected': response.status == 200,
                                'status_code': response.status
                            }
                except Exception as e:
                    services['snowflake_cortex'] = {
                        'connected': False,
                        'error': str(e)
                    }
            
            return services
            
        except Exception as e:
            logger.error(f"External services health check failed: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_status(self, results: Dict[str, Any]) -> str:
        """
        Calculate overall system health status
        Learning Outcome: System health aggregation
        """
        try:
            total_checks = len(results)
            healthy_checks = sum(1 for result in results.values() 
                               if result.get('status') == 'healthy')
            
            if healthy_checks == total_checks:
                return 'healthy'
            elif healthy_checks >= total_checks * 0.8:
                return 'degraded'
            else:
                return 'unhealthy'
                
        except Exception as e:
            logger.error(f"Error calculating overall status: {e}")
            return 'unknown' 