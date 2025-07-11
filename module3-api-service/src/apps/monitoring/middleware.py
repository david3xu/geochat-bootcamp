"""
Performance monitoring middleware
Full Stack AI Engineer Bootcamp - Module 3
"""

import time
import logging
from django.utils.deprecation import MiddlewareMixin
from django.conf import settings

logger = logging.getLogger('chat2map')

class PerformanceMonitoringMiddleware(MiddlewareMixin):
    """
    Middleware for monitoring API performance
    Learning Outcome: Request/response performance tracking
    """
    
    def process_request(self, request):
        """Start timing the request"""
        request.start_time = time.time()
        return None
    
    def process_response(self, request, response):
        """Calculate and log response time"""
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            
            # Log performance metrics
            self._log_performance_metrics(request, response, duration)
            
            # Add performance headers
            response['X-Response-Time'] = f"{duration:.3f}s"
            response['X-Request-ID'] = getattr(request, 'request_id', 'unknown')
        
        return response
    
    def process_exception(self, request, exception):
        """Handle exceptions and log performance"""
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            
            # Log error with performance data
            logger.error(
                f"Request failed: {request.path} - {duration:.3f}s - {str(exception)}"
            )
        
        return None
    
    def _log_performance_metrics(self, request, response, duration):
        """Log performance metrics for monitoring"""
        try:
            # Only log if performance monitoring is enabled
            if not getattr(settings, 'PERFORMANCE_MONITORING_ENABLED', False):
                return
            
            # Extract relevant information
            path = request.path
            method = request.method
            status_code = response.status_code
            user_agent = request.META.get('HTTP_USER_AGENT', 'unknown')
            
            # Log performance data
            logger.info(
                f"Performance: {method} {path} - {status_code} - {duration:.3f}s"
            )
            
            # Log slow requests
            if duration > 1.0:  # More than 1 second
                logger.warning(
                    f"Slow request: {method} {path} - {duration:.3f}s"
                )
            
            # Log very slow requests
            if duration > 5.0:  # More than 5 seconds
                logger.error(
                    f"Very slow request: {method} {path} - {duration:.3f}s"
                )
                
        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}") 