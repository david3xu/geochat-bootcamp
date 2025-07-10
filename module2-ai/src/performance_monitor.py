"""
AI Performance Monitoring and Optimization
Measurable Success: Real-time performance tracking with <2s response time targets
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import psutil
import threading
from datetime import datetime, timedelta
import json

from .config import config

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive AI performance metrics"""
    timestamp: str
    cortex_embed_calls: int
    cortex_complete_calls: int
    average_embed_time_ms: float
    average_complete_time_ms: float
    vector_search_queries: int
    average_search_time_ms: float
    qa_queries: int
    average_qa_time_ms: float
    system_cpu_percent: float
    system_memory_percent: float
    error_rate_percentage: float
    performance_target_compliance: float

@dataclass
class AlertThreshold:
    """Performance alert thresholds"""
    max_embed_time_ms: float
    max_complete_time_ms: float
    max_search_time_ms: float
    max_qa_time_ms: float
    max_cpu_percent: float
    max_memory_percent: float
    max_error_rate_percent: float

class AIPerformanceMonitor:
    """
    Real-time AI performance monitoring and alerting
    Measurable Success: <2s response time with 95%+ target compliance
    """
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_thresholds = AlertThreshold(
            max_embed_time_ms=500,
            max_complete_time_ms=2000,
            max_search_time_ms=100,
            max_qa_time_ms=2000,
            max_cpu_percent=80.0,
            max_memory_percent=80.0,
            max_error_rate_percent=5.0
        )
        self.current_metrics = self._initialize_metrics()
        self.monitoring_active = False
        self.monitor_thread = None
    
    def _initialize_metrics(self) -> PerformanceMetrics:
        """Initialize performance metrics"""
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            cortex_embed_calls=0,
            cortex_complete_calls=0,
            average_embed_time_ms=0.0,
            average_complete_time_ms=0.0,
            vector_search_queries=0,
            average_search_time_ms=0.0,
            qa_queries=0,
            average_qa_time_ms=0.0,
            system_cpu_percent=0.0,
            system_memory_percent=0.0,
            error_rate_percentage=0.0,
            performance_target_compliance=0.0
        )
    
    def start_monitoring(self, interval_seconds: int = 30) -> bool:
        """
        Start continuous performance monitoring
        Success Metric: Real-time monitoring with <1s overhead
        """
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return False
        
        try:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval_seconds,),
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info(f"✅ Performance monitoring started with {interval_seconds}s interval")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {str(e)}")
            self.monitoring_active = False
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("✅ Performance monitoring stopped")
        return True
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Calculate performance compliance
                self._calculate_performance_compliance()
                
                # Store metrics
                self.metrics_history.append(self.current_metrics)
                
                # Check for alerts
                self._check_performance_alerts()
                
                # Clean old metrics (keep last 24 hours)
                self._cleanup_old_metrics()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(interval_seconds)
    
    def _update_system_metrics(self) -> None:
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Update current metrics
            self.current_metrics.system_cpu_percent = cpu_percent
            self.current_metrics.system_memory_percent = memory_percent
            self.current_metrics.timestamp = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {str(e)}")
    
    def _calculate_performance_compliance(self) -> None:
        """Calculate overall performance target compliance"""
        compliance_checks = [
            self.current_metrics.average_embed_time_ms <= self.alert_thresholds.max_embed_time_ms,
            self.current_metrics.average_complete_time_ms <= self.alert_thresholds.max_complete_time_ms,
            self.current_metrics.average_search_time_ms <= self.alert_thresholds.max_search_time_ms,
            self.current_metrics.average_qa_time_ms <= self.alert_thresholds.max_qa_time_ms,
            self.current_metrics.system_cpu_percent <= self.alert_thresholds.max_cpu_percent,
            self.current_metrics.system_memory_percent <= self.alert_thresholds.max_memory_percent,
            self.current_metrics.error_rate_percentage <= self.alert_thresholds.max_error_rate_percent
        ]
        
        compliance_percentage = (sum(compliance_checks) / len(compliance_checks)) * 100
        self.current_metrics.performance_target_compliance = compliance_percentage
    
    def _check_performance_alerts(self) -> None:
        """Check for performance alerts and log warnings"""
        alerts = []
        
        if self.current_metrics.average_embed_time_ms > self.alert_thresholds.max_embed_time_ms:
            alerts.append(f"Embed time {self.current_metrics.average_embed_time_ms:.2f}ms exceeds {self.alert_thresholds.max_embed_time_ms}ms")
        
        if self.current_metrics.average_complete_time_ms > self.alert_thresholds.max_complete_time_ms:
            alerts.append(f"Complete time {self.current_metrics.average_complete_time_ms:.2f}ms exceeds {self.alert_thresholds.max_complete_time_ms}ms")
        
        if self.current_metrics.average_search_time_ms > self.alert_thresholds.max_search_time_ms:
            alerts.append(f"Search time {self.current_metrics.average_search_time_ms:.2f}ms exceeds {self.alert_thresholds.max_search_time_ms}ms")
        
        if self.current_metrics.average_qa_time_ms > self.alert_thresholds.max_qa_time_ms:
            alerts.append(f"QA time {self.current_metrics.average_qa_time_ms:.2f}ms exceeds {self.alert_thresholds.max_qa_time_ms}ms")
        
        if self.current_metrics.system_cpu_percent > self.alert_thresholds.max_cpu_percent:
            alerts.append(f"CPU usage {self.current_metrics.system_cpu_percent:.1f}% exceeds {self.alert_thresholds.max_cpu_percent}%")
        
        if self.current_metrics.system_memory_percent > self.alert_thresholds.max_memory_percent:
            alerts.append(f"Memory usage {self.current_metrics.system_memory_percent:.1f}% exceeds {self.alert_thresholds.max_memory_percent}%")
        
        if self.current_metrics.error_rate_percentage > self.alert_thresholds.max_error_rate_percent:
            alerts.append(f"Error rate {self.current_metrics.error_rate_percentage:.1f}% exceeds {self.alert_thresholds.max_error_rate_percent}%")
        
        if alerts:
            logger.warning(f"Performance alerts: {'; '.join(alerts)}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than 24 hours"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics.timestamp) > cutoff_time
        ]
    
    def record_cortex_embed_call(self, processing_time_ms: float, success: bool) -> None:
        """Record Cortex embedding function call"""
        self.current_metrics.cortex_embed_calls += 1
        
        # Update average time
        n = self.current_metrics.cortex_embed_calls
        self.current_metrics.average_embed_time_ms = (
            (self.current_metrics.average_embed_time_ms * (n - 1) + processing_time_ms) / n
        )
        
        # Update error rate
        if not success:
            self._update_error_rate()
    
    def record_cortex_complete_call(self, processing_time_ms: float, success: bool) -> None:
        """Record Cortex completion function call"""
        self.current_metrics.cortex_complete_calls += 1
        
        # Update average time
        n = self.current_metrics.cortex_complete_calls
        self.current_metrics.average_complete_time_ms = (
            (self.current_metrics.average_complete_time_ms * (n - 1) + processing_time_ms) / n
        )
        
        # Update error rate
        if not success:
            self._update_error_rate()
    
    def record_vector_search_query(self, processing_time_ms: float, success: bool) -> None:
        """Record vector search query"""
        self.current_metrics.vector_search_queries += 1
        
        # Update average time
        n = self.current_metrics.vector_search_queries
        self.current_metrics.average_search_time_ms = (
            (self.current_metrics.average_search_time_ms * (n - 1) + processing_time_ms) / n
        )
        
        # Update error rate
        if not success:
            self._update_error_rate()
    
    def record_qa_query(self, processing_time_ms: float, success: bool) -> None:
        """Record QA query"""
        self.current_metrics.qa_queries += 1
        
        # Update average time
        n = self.current_metrics.qa_queries
        self.current_metrics.average_qa_time_ms = (
            (self.current_metrics.average_qa_time_ms * (n - 1) + processing_time_ms) / n
        )
        
        # Update error rate
        if not success:
            self._update_error_rate()
    
    def _update_error_rate(self) -> None:
        """Update error rate calculation"""
        total_calls = (
            self.current_metrics.cortex_embed_calls +
            self.current_metrics.cortex_complete_calls +
            self.current_metrics.vector_search_queries +
            self.current_metrics.qa_queries
        )
        
        if total_calls > 0:
            # Simplified error rate calculation
            self.current_metrics.error_rate_percentage = min(
                self.current_metrics.error_rate_percentage + 1.0, 100.0
            )
    
    def get_current_performance_report(self) -> Dict[str, Any]:
        """Get current performance report for supervision"""
        return {
            'current_metrics': asdict(self.current_metrics),
            'alert_thresholds': asdict(self.alert_thresholds),
            'performance_assessment': {
                'embed_performance_target_met': self.current_metrics.average_embed_time_ms <= self.alert_thresholds.max_embed_time_ms,
                'complete_performance_target_met': self.current_metrics.average_complete_time_ms <= self.alert_thresholds.max_complete_time_ms,
                'search_performance_target_met': self.current_metrics.average_search_time_ms <= self.alert_thresholds.max_search_time_ms,
                'qa_performance_target_met': self.current_metrics.average_qa_time_ms <= self.alert_thresholds.max_qa_time_ms,
                'system_performance_target_met': (
                    self.current_metrics.system_cpu_percent <= self.alert_thresholds.max_cpu_percent and
                    self.current_metrics.system_memory_percent <= self.alert_thresholds.max_memory_percent
                ),
                'error_rate_target_met': self.current_metrics.error_rate_percentage <= self.alert_thresholds.max_error_rate_percent,
                'overall_compliance_percentage': self.current_metrics.performance_target_compliance
            },
            'learning_targets_assessment': {
                'cortex_usage_target_1000_calls': (
                    self.current_metrics.cortex_embed_calls + self.current_metrics.cortex_complete_calls >= 1000
                ),
                'performance_target_2s_response': self.current_metrics.average_complete_time_ms <= 2000,
                'quality_target_85_percent': self.current_metrics.performance_target_compliance >= 85.0
            },
            'system_health': {
                'monitoring_active': self.monitoring_active,
                'metrics_history_length': len(self.metrics_history),
                'last_update': self.current_metrics.timestamp
            }
        }
    
    def get_historical_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical performance report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics.timestamp) > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No historical metrics available'}
        
        # Calculate historical averages
        avg_embed_time = sum(m.average_embed_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_complete_time = sum(m.average_complete_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_search_time = sum(m.average_search_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_qa_time = sum(m.average_qa_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_compliance = sum(m.performance_target_compliance for m in recent_metrics) / len(recent_metrics)
        
        # Calculate totals
        total_embed_calls = sum(m.cortex_embed_calls for m in recent_metrics)
        total_complete_calls = sum(m.cortex_complete_calls for m in recent_metrics)
        total_search_queries = sum(m.vector_search_queries for m in recent_metrics)
        total_qa_queries = sum(m.qa_queries for m in recent_metrics)
        
        return {
            'time_period_hours': hours,
            'metrics_count': len(recent_metrics),
            'historical_averages': {
                'average_embed_time_ms': avg_embed_time,
                'average_complete_time_ms': avg_complete_time,
                'average_search_time_ms': avg_search_time,
                'average_qa_time_ms': avg_qa_time,
                'average_compliance_percentage': avg_compliance
            },
            'total_activity': {
                'total_embed_calls': total_embed_calls,
                'total_complete_calls': total_complete_calls,
                'total_search_queries': total_search_queries,
                'total_qa_queries': total_qa_queries,
                'total_cortex_calls': total_embed_calls + total_complete_calls
            },
            'performance_trends': {
                'embed_performance_trend': self._calculate_trend([m.average_embed_time_ms for m in recent_metrics]),
                'complete_performance_trend': self._calculate_trend([m.average_complete_time_ms for m in recent_metrics]),
                'search_performance_trend': self._calculate_trend([m.average_search_time_ms for m in recent_metrics]),
                'qa_performance_trend': self._calculate_trend([m.average_qa_time_ms for m in recent_metrics])
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend (improving, stable, declining)"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        change_percent = ((avg_second - avg_first) / avg_first) * 100 if avg_first > 0 else 0
        
        if change_percent < -5:
            return 'improving'
        elif change_percent > 5:
            return 'declining'
        else:
            return 'stable'
    
    def export_performance_data(self, filepath: str) -> bool:
        """Export performance data to JSON file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'current_metrics': asdict(self.current_metrics),
                'metrics_history': [asdict(m) for m in self.metrics_history],
                'alert_thresholds': asdict(self.alert_thresholds)
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"✅ Performance data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {str(e)}")
            return False 