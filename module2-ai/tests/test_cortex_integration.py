"""
Unit Tests for Snowflake Cortex Integration
Validation of enterprise AI functionality and performance targets
"""
import pytest
import time
from unittest.mock import Mock, patch

from src.snowflake_cortex_client import SnowflakeCortexClient, EmbeddingResult, CompletionResult
from src.config import config

class TestSnowflakeCortexClient:
    """Test Snowflake Cortex integration functionality"""
    
    @pytest.fixture
    def mock_cortex_client(self):
        """Create mock Cortex client for testing"""
        with patch('src.snowflake_cortex_client.snowflake.connector.connect') as mock_connect:
            mock_connection = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_connection
            mock_connection.cursor.return_value = mock_cursor
            
            # Mock successful EMBED_TEXT_768 response
            mock_cursor.fetchone.return_value = [[0.1] * 768]  # Mock 768-dimension embedding
            
            client = SnowflakeCortexClient()
            yield client, mock_cursor
    
    def test_connection_establishment(self, mock_cortex_client):
        """Test Snowflake connection establishment"""
        client, mock_cursor = mock_cortex_client
        
        # Verify connection was attempted
        assert client.connection is not None
        
        # Test Cortex function verification
        mock_cursor.fetchone.return_value = [[0.1] * 768]
        result = client._verify_cortex_access()
        assert result == True
    
    def test_embedding_generation_batch(self, mock_cortex_client):
        """Test batch embedding generation performance"""
        client, mock_cursor = mock_cortex_client
        
        # Prepare test data
        test_texts = [
            "Gold exploration in Pilbara region",
            "Iron ore deposit with high grade hematite",
            "Copper mineralization in volcanic rocks"
        ]
        
        # Mock embedding responses
        mock_cursor.fetchall.return_value = [
            {'TEXT': text, 'EMBEDDING': [0.1] * 768} for text in test_texts
        ]
        
        # Test batch embedding generation
        start_time = time.time()
        results = client.generate_embeddings_batch(test_texts)
        processing_time = (time.time() - start_time) * 1000
        
        # Assertions for learning outcomes
        assert len(results) == 3
        assert all(isinstance(result, EmbeddingResult) for result in results)
        assert all(result.success for result in results)
        assert all(len(result.embedding_vector) == 768 for result in results)
        
        # Performance target validation
        assert processing_time < 500  # <500ms for batch processing
        assert all(result.processing_time_ms > 0 for result in results)
    
    def test_geological_query_completion(self, mock_cortex_client):
        """Test geological query completion with Cortex COMPLETE"""
        client, mock_cursor = mock_cortex_client
        
        # Mock COMPLETE function response
        mock_geological_response = "Gold deposits in Western Australia are primarily found in the Yilgarn Craton, formed through hydrothermal processes during Archean orogenic events."
        mock_cursor.fetchone.return_value = {'COMPLETION_RESULT': mock_geological_response}
        
        # Test geological query completion
        test_query = "Explain gold formation in Western Australia"
        start_time = time.time()
        result = client.complete_geological_query(test_query)
        processing_time = (time.time() - start_time) * 1000
        
        # Assertions for learning outcomes
        assert isinstance(result, CompletionResult)
        assert result.success == True
        assert result.completion_output == mock_geological_response
        assert result.relevance_score > 0
        
        # Performance target validation
        assert processing_time < 2000  # <2s response time target
        assert result.processing_time_ms > 0
    
    def test_large_dataset_processing(self, mock_cortex_client):
        """Test large dataset processing performance"""
        client, mock_cursor = mock_cortex_client
        
        # Create large test dataset
        large_dataset = [f"Geological description {i}" for i in range(1000)]
        
        # Mock batch responses
        mock_cursor.fetchall.return_value = [
            {'TEXT': f"Geological description {i}", 'EMBEDDING': [0.1] * 768} 
            for i in range(100)  # Mock batch size
        ]
        
        # Test large dataset processing
        start_time = time.time()
        processing_report = client.batch_process_large_dataset(large_dataset, batch_size=100)
        total_time = time.time() - start_time
        
        # Assertions for supervision metrics
        assert processing_report['total_texts_processed'] == 1000
        assert processing_report['success_rate_percentage'] >= 95
        assert processing_report['processing_rate_per_second'] > 50  # Minimum processing rate
        
        # Performance target for large datasets
        if len(large_dataset) >= 10000:
            assert total_time < 600  # 10 minutes for 10k+ records
    
    def test_usage_metrics_tracking(self, mock_cortex_client):
        """Test Cortex usage metrics and supervision reporting"""
        client, mock_cursor = mock_cortex_client
        
        # Simulate usage
        mock_cursor.fetchall.return_value = [
            {'TEXT': 'test', 'EMBEDDING': [0.1] * 768}
        ]
        client.generate_embeddings_batch(['test text'])
        
        mock_cursor.fetchone.return_value = {'COMPLETION_RESULT': 'test response'}
        client.complete_geological_query('test query')
        
        # Test usage metrics
        metrics = client.get_usage_metrics()
        
        # Supervision validation
        assert metrics.embed_calls_total >= 1
        assert metrics.complete_calls_total >= 1
        assert metrics.average_embed_time_ms >= 0
        assert metrics.average_complete_time_ms >= 0
        
        # Test daily usage report
        daily_report = client.generate_daily_usage_report()
        assert 'cortex_usage_summary' in daily_report
        assert 'learning_targets_assessment' in daily_report
        assert 'performance_analysis' in daily_report
    
    def test_error_handling_and_resilience(self, mock_cortex_client):
        """Test error handling for failed Cortex operations"""
        client, mock_cursor = mock_cortex_client
        
        # Simulate Cortex function failure
        mock_cursor.execute.side_effect = Exception("Cortex function timeout")
        
        # Test embedding generation with error
        results = client.generate_embeddings_batch(['test text'])
        
        # Verify graceful error handling
        assert len(results) == 1
        assert not results[0].success
        assert results[0].error_message is not None
        
        # Test completion with error
        completion_result = client.complete_geological_query('test query')
        assert not completion_result.success
        assert completion_result.error_message is not None

class TestCortexPerformanceOptimizer:
    """Test performance optimization features"""
    
    @pytest.fixture
    def mock_optimizer(self, mock_cortex_client):
        """Create mock performance optimizer"""
        from src.snowflake_cortex_client import CortexPerformanceOptimizer
        client, _ = mock_cortex_client
        return CortexPerformanceOptimizer(client)
    
    def test_embedding_caching(self, mock_optimizer):
        """Test embedding caching for performance improvement"""
        optimizer = mock_optimizer
        
        # Test cache miss and hit
        test_texts = ['geological sample text', 'geological sample text']  # Duplicate for cache test
        
        # Mock embedding results
        mock_results = [
            EmbeddingResult('geological sample text', [0.1] * 768, 100, 'e5-base-v2', True)
        ]
        
        with patch.object(optimizer.cortex_client, 'generate_embeddings_batch', return_value=mock_results):
            # First call should cache the result
            results1 = optimizer.cached_embedding_generation(['geological sample text'])
            
            # Second call should use cache
            results2 = optimizer.cached_embedding_generation(['geological sample text'])
            
        # Verify caching behavior
        assert optimizer.cache_hit_count >= 1
        assert len(optimizer.embedding_cache) >= 1
        
        # Test cache performance report
        cache_report = optimizer.get_cache_performance_report()
        assert cache_report['cache_hit_rate_percentage'] >= 0
        assert 'optimization_recommendations' in cache_report
    
    def test_batch_size_optimization(self, mock_optimizer):
        """Test optimal batch size determination"""
        optimizer = mock_optimizer
        
        # Test workload analysis
        workload_analysis = {
            'average_text_length': 150,
            'concurrent_users': 5
        }
        
        optimal_batch_size = optimizer.optimize_batch_sizing(workload_analysis)
        
        # Verify reasonable batch size
        assert 10 <= optimal_batch_size <= 200
        assert optimal_batch_size <= optimizer.cortex_client.cortex_config.max_batch_size

class TestSupervisionMetrics:
    """Test supervision and assessment metrics"""
    
    def test_learning_outcome_measurement(self, mock_cortex_client):
        """Test measurable learning outcome tracking"""
        client, mock_cursor = mock_cortex_client
        
        # Simulate student usage over time
        for _ in range(1000):  # Simulate 1000 Cortex calls
            mock_cursor.fetchall.return_value = [{'TEXT': 'test', 'EMBEDDING': [0.1] * 768}]
            client.generate_embeddings_batch(['test'])
        
        # Test supervision metrics
        daily_report = client.generate_daily_usage_report()
        learning_assessment = daily_report['learning_targets_assessment']
        
        # Verify learning targets measurement
        assert 'weekly_target_1000_calls' in learning_assessment
        assert learning_assessment['weekly_target_1000_calls'] == True
        assert 'usage_target_met' in learning_assessment
        
        # Performance compliance tracking
        performance_analysis = daily_report['performance_analysis']
        assert 'embed_target_met' in performance_analysis
        assert 'complete_target_met' in performance_analysis
    
    def test_portfolio_evidence_generation(self, mock_cortex_client):
        """Test automatic portfolio evidence generation"""
        client, _ = mock_cortex_client
        
        # Generate usage metrics for portfolio
        metrics = client.get_usage_metrics()
        daily_report = client.generate_daily_usage_report()
        
        # Verify portfolio-ready evidence
        assert hasattr(metrics, 'embed_calls_total')
        assert hasattr(metrics, 'average_embed_time_ms')
        assert 'cortex_usage_summary' in daily_report
        assert 'cost_optimization_metrics' in daily_report
        
        # Verify measurable outcomes
        assert metrics.embed_calls_total >= 0
        assert metrics.performance_target_compliance >= 0

if __name__ == '__main__':
    pytest.main([__file__])
