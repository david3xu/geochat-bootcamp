"""
Performance Monitor Validation Tests
Validation of AI performance tracking and optimization
"""
import pytest
import time
from unittest.mock import Mock, patch

from src.performance_monitor import AIPerformanceMonitor, PerformanceMetrics

class TestAIPerformanceMonitor:
    """Test AI performance monitoring functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return AIPerformanceMonitor()
    
    def test_performance_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization"""
        assert performance_monitor.metrics_history is not None
        assert performance_monitor.alert_thresholds is not None
        assert performance_monitor.current_metrics is not None
        assert performance_monitor.monitoring_active == False
    
    def test_metrics_initialization(self, performance_monitor):
        """Test performance metrics initialization"""
        metrics = performance_monitor.current_metrics
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.cortex_embed_calls == 0
        assert metrics.cortex_complete_calls == 0
        assert metrics.average_embed_time_ms == 0.0
        assert metrics.average_complete_time_ms == 0.0
        assert metrics.vector_search_queries == 0
        assert metrics.average_search_time_ms == 0.0
        assert metrics.qa_queries == 0
        assert metrics.average_qa_time_ms == 0.0
        assert metrics.system_cpu_percent == 0.0
        assert metrics.system_memory_percent == 0.0
        assert metrics.error_rate_percentage == 0.0
        assert metrics.performance_target_compliance == 0.0
    
    def test_cortex_embed_call_recording(self, performance_monitor):
        """Test recording of Cortex embedding calls"""
        # Record successful embed call
        performance_monitor.record_cortex_embed_call(150.0, True)
        
        metrics = performance_monitor.current_metrics
        assert metrics.cortex_embed_calls == 1
        assert metrics.average_embed_time_ms == 150.0
        
        # Record failed embed call
        performance_monitor.record_cortex_embed_call(200.0, False)
        
        metrics = performance_monitor.current_metrics
        assert metrics.cortex_embed_calls == 2
        assert metrics.average_embed_time_ms == 175.0  # Average of 150 and 200
    
    def test_cortex_complete_call_recording(self, performance_monitor):
        """Test recording of Cortex completion calls"""
        # Record successful complete call
        performance_monitor.record_cortex_complete_call(1800.0, True)
        
        metrics = performance_monitor.current_metrics
        assert metrics.cortex_complete_calls == 1
        assert metrics.average_complete_time_ms == 1800.0
        
        # Record failed complete call
        performance_monitor.record_cortex_complete_call(2200.0, False)
        
        metrics = performance_monitor.current_metrics
        assert metrics.cortex_complete_calls == 2
        assert metrics.average_complete_time_ms == 2000.0  # Average of 1800 and 2200
    
    def test_vector_search_query_recording(self, performance_monitor):
        """Test recording of vector search queries"""
        # Record successful search query
        performance_monitor.record_vector_search_query(85.0, True)
        
        metrics = performance_monitor.current_metrics
        assert metrics.vector_search_queries == 1
        assert metrics.average_search_time_ms == 85.0
        
        # Record failed search query
        performance_monitor.record_vector_search_query(120.0, False)
        
        metrics = performance_monitor.current_metrics
        assert metrics.vector_search_queries == 2
        assert metrics.average_search_time_ms == 102.5  # Average of 85 and 120
    
    def test_qa_query_recording(self, performance_monitor):
        """Test recording of QA queries"""
        # Record successful QA query
        performance_monitor.record_qa_query(1600.0, True)
        
        metrics = performance_monitor.current_metrics
        assert metrics.qa_queries == 1
        assert metrics.average_qa_time_ms == 1600.0
        
        # Record failed QA query
        performance_monitor.record_qa_query(1900.0, False)
        
        metrics = performance_monitor.current_metrics
        assert metrics.qa_queries == 2
        assert metrics.average_qa_time_ms == 1750.0  # Average of 1600 and 1900
    
    def test_error_rate_calculation(self, performance_monitor):
        """Test error rate calculation"""
        # Record some successful calls
        performance_monitor.record_cortex_embed_call(100.0, True)
        performance_monitor.record_cortex_complete_call(1500.0, True)
        performance_monitor.record_vector_search_query(80.0, True)
        performance_monitor.record_qa_query(1600.0, True)
        
        # Record some failed calls
        performance_monitor.record_cortex_embed_call(200.0, False)
        performance_monitor.record_cortex_complete_call(2000.0, False)
        
        metrics = performance_monitor.current_metrics
        assert metrics.error_rate_percentage > 0
        assert metrics.error_rate_percentage <= 100.0
    
    def test_performance_compliance_calculation(self, performance_monitor):
        """Test performance compliance calculation"""
        # Set up metrics that meet targets
        performance_monitor.current_metrics.average_embed_time_ms = 400  # < 500ms target
        performance_monitor.current_metrics.average_complete_time_ms = 1800  # < 2000ms target
        performance_monitor.current_metrics.average_search_time_ms = 80  # < 100ms target
        performance_monitor.current_metrics.average_qa_time_ms = 1800  # < 2000ms target
        performance_monitor.current_metrics.system_cpu_percent = 60  # < 80% target
        performance_monitor.current_metrics.system_memory_percent = 70  # < 80% target
        performance_monitor.current_metrics.error_rate_percentage = 3  # < 5% target
        
        performance_monitor._calculate_performance_compliance()
        
        # Should have 100% compliance
        assert performance_monitor.current_metrics.performance_target_compliance == 100.0
    
    def test_performance_alert_generation(self, performance_monitor):
        """Test performance alert generation"""
        # Set up metrics that exceed thresholds
        performance_monitor.current_metrics.average_embed_time_ms = 600  # > 500ms threshold
        performance_monitor.current_metrics.average_complete_time_ms = 2500  # > 2000ms threshold
        performance_monitor.current_metrics.system_cpu_percent = 85  # > 80% threshold
        performance_monitor.current_metrics.error_rate_percentage = 8  # > 5% threshold
        
        # Capture log messages
        with patch('logging.Logger.warning') as mock_warning:
            performance_monitor._check_performance_alerts()
            
            # Verify alerts were generated
            assert mock_warning.call_count >= 4  # At least 4 alerts
    
    def test_current_performance_report(self, performance_monitor):
        """Test current performance report generation"""
        # Record some activity
        performance_monitor.record_cortex_embed_call(150.0, True)
        performance_monitor.record_cortex_complete_call(1800.0, True)
        performance_monitor.record_vector_search_query(85.0, True)
        performance_monitor.record_qa_query(1600.0, True)
        
        report = performance_monitor.get_current_performance_report()
        
        # Verify report structure
        assert 'current_metrics' in report
        assert 'alert_thresholds' in report
        assert 'performance_assessment' in report
        assert 'learning_targets_assessment' in report
        assert 'system_health' in report
        
        # Verify metrics
        metrics = report['current_metrics']
        assert metrics['cortex_embed_calls'] == 1
        assert metrics['cortex_complete_calls'] == 1
        assert metrics['vector_search_queries'] == 1
        assert metrics['qa_queries'] == 1
    
    def test_historical_performance_report(self, performance_monitor):
        """Test historical performance report generation"""
        # Add some historical metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=f"2024-01-{i+1:02d}T10:00:00",
                cortex_embed_calls=i+1,
                cortex_complete_calls=i+1,
                average_embed_time_ms=100.0 + i*10,
                average_complete_time_ms=1500.0 + i*50,
                vector_search_queries=i+1,
                average_search_time_ms=80.0 + i*5,
                qa_queries=i+1,
                average_qa_time_ms=1600.0 + i*30,
                system_cpu_percent=50.0 + i*5,
                system_memory_percent=60.0 + i*5,
                error_rate_percentage=2.0 + i*0.5,
                performance_target_compliance=90.0 + i*2
            )
            performance_monitor.metrics_history.append(metrics)
        
        report = performance_monitor.get_historical_performance_report(hours=24)
        
        # Verify report structure
        assert 'time_period_hours' in report
        assert 'metrics_count' in report
        assert 'historical_averages' in report
        assert 'total_activity' in report
        assert 'performance_trends' in report
        
        # Verify data
        assert report['metrics_count'] == 5
        assert report['historical_averages']['average_embed_time_ms'] > 0
        assert report['total_activity']['total_cortex_calls'] == 15  # Sum of 1+2+3+4+5
    
    def test_trend_calculation(self, performance_monitor):
        """Test performance trend calculation"""
        # Test improving trend
        improving_values = [100, 90, 80, 70, 60]
        trend = performance_monitor._calculate_trend(improving_values)
        assert trend == 'improving'
        
        # Test declining trend
        declining_values = [60, 70, 80, 90, 100]
        trend = performance_monitor._calculate_trend(declining_values)
        assert trend == 'declining'
        
        # Test stable trend
        stable_values = [80, 82, 78, 81, 79]
        trend = performance_monitor._calculate_trend(stable_values)
        assert trend == 'stable'
        
        # Test insufficient data
        insufficient_values = [80]
        trend = performance_monitor._calculate_trend(insufficient_values)
        assert trend == 'insufficient_data'
    
    def test_performance_data_export(self, performance_monitor, tmp_path):
        """Test performance data export"""
        # Record some activity
        performance_monitor.record_cortex_embed_call(150.0, True)
        performance_monitor.record_cortex_complete_call(1800.0, True)
        
        # Export data
        export_file = tmp_path / "performance_data.json"
        success = performance_monitor.export_performance_data(str(export_file))
        
        assert success == True
        assert export_file.exists()
        
        # Verify exported data
        import json
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        assert 'export_timestamp' in data
        assert 'current_metrics' in data
        assert 'metrics_history' in data
        assert 'alert_thresholds' in data

class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing"""
        return AIPerformanceMonitor()
    
    def test_monitoring_start_stop(self, performance_monitor):
        """Test monitoring start and stop functionality"""
        # Start monitoring
        success = performance_monitor.start_monitoring(interval_seconds=1)
        assert success == True
        assert performance_monitor.monitoring_active == True
        
        # Try to start again (should fail)
        success = performance_monitor.start_monitoring(interval_seconds=1)
        assert success == False
        
        # Stop monitoring
        success = performance_monitor.stop_monitoring()
        assert success == True
        assert performance_monitor.monitoring_active == False
    
    def test_system_metrics_update(self, performance_monitor):
        """Test system metrics update"""
        # Mock psutil for testing
        with patch('psutil.cpu_percent', return_value=65.5):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 72.3
                
                performance_monitor._update_system_metrics()
                
                metrics = performance_monitor.current_metrics
                assert metrics.system_cpu_percent == 65.5
                assert metrics.system_memory_percent == 72.3
    
    def test_metrics_history_cleanup(self, performance_monitor):
        """Test metrics history cleanup"""
        # Add old metrics (more than 24 hours ago)
        from datetime import datetime, timedelta
        
        old_timestamp = (datetime.now() - timedelta(hours=25)).isoformat()
        old_metrics = PerformanceMetrics(
            timestamp=old_timestamp,
            cortex_embed_calls=1,
            cortex_complete_calls=1,
            average_embed_time_ms=100.0,
            average_complete_time_ms=1500.0,
            vector_search_queries=1,
            average_search_time_ms=80.0,
            qa_queries=1,
            average_qa_time_ms=1600.0,
            system_cpu_percent=50.0,
            system_memory_percent=60.0,
            error_rate_percentage=2.0,
            performance_target_compliance=90.0
        )
        
        performance_monitor.metrics_history.append(old_metrics)
        initial_count = len(performance_monitor.metrics_history)
        
        # Add recent metrics
        recent_timestamp = datetime.now().isoformat()
        recent_metrics = PerformanceMetrics(
            timestamp=recent_timestamp,
            cortex_embed_calls=2,
            cortex_complete_calls=2,
            average_embed_time_ms=110.0,
            average_complete_time_ms=1550.0,
            vector_search_queries=2,
            average_search_time_ms=85.0,
            qa_queries=2,
            average_qa_time_ms=1650.0,
            system_cpu_percent=55.0,
            system_memory_percent=65.0,
            error_rate_percentage=2.5,
            performance_target_compliance=92.0
        )
        
        performance_monitor.metrics_history.append(recent_metrics)
        
        # Trigger cleanup
        performance_monitor._cleanup_old_metrics()
        
        # Verify old metrics were removed
        assert len(performance_monitor.metrics_history) < initial_count + 1
        assert all(
            datetime.fromisoformat(metrics.timestamp) > datetime.now() - timedelta(hours=24)
            for metrics in performance_monitor.metrics_history
        )

if __name__ == '__main__':
    pytest.main([__file__]) 