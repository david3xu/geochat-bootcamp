"""
Health monitor for integration services
Full Stack AI Engineer Bootcamp - Module 3
"""

import asyncio
import logging
from typing import Dict, Any
from django.conf import settings

logger = logging.getLogger('chat2map')

class IntegrationHealthMonitor:
    """
    Health monitor for integration services
    Learning Outcome: Service health monitoring and alerting
    """
    
    def __init__(self):
        self.services = {
            'module1': settings.MODULE1_API_URL,
            'module2': settings.MODULE2_AI_URL,
            'snowflake_cortex': settings.SNOWFLAKE_CORTEX_URL
        }
    
    async def check_all_services(self) -> Dict[str, Any]:
        """
        Check health of all integration services
        Learning Outcome: Comprehensive service monitoring
        """
        results = {}
        
        for service_name, service_url in self.services.items():
            if service_url:
                health_status = await self._check_service_health(service_name, service_url)
                results[service_name] = health_status
        
        return results
    
    async def _check_service_health(self, service_name: str, service_url: str) -> Dict[str, Any]:
        """
        Check health of a specific service
        Learning Outcome: Individual service health monitoring
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{service_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'healthy',
                            'response_time': data.get('response_time', 0),
                            'details': data
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'status_code': response.status,
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            } 