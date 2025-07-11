"""
Vector Database Operation Tests
Validation of vector storage and similarity search functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import time

from src.vector_database import VectorDatabaseManager, VectorRecord
from src.semantic_search import GeologicalSemanticSearch, SimilarityMatch

class TestVectorDatabaseManager:
    """Test vector database operations"""
    
    @pytest.fixture
    def mock_cosmos_client(self):
        """Create mock Cosmos DB client"""
        with patch('src.vector_database.CosmosClient') as mock_client:
            mock_container = Mock()
            mock_database = Mock()
            mock_client.from_connection_string.return_value = mock_client
            mock_client.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_container
            
            # Mock container read (exists)
            mock_container.read.return_value = {}
            
            yield mock_client, mock_container
    
    @pytest.fixture
    def vector_db_manager(self, mock_cosmos_client):
        """Create vector database manager with mock client"""
        mock_client, mock_container = mock_cosmos_client
        return VectorDatabaseManager()
    
    def test_database_initialization(self, vector_db_manager, mock_cosmos_client):
        """Test Cosmos DB initialization"""
        mock_client, mock_container = mock_cosmos_client
        
        # Verify initialization
        assert vector_db_manager.client is not None
        assert vector_db_manager.database is not None
        assert vector_db_manager.container is not None
    
    def test_vector_storage(self, vector_db_manager, mock_cosmos_client):
        """Test geological vector storage"""
        mock_client, mock_container = mock_cosmos_client
        
        # Create test vector records
        test_vectors = [
            VectorRecord(
                id="test_1",
                partition_key="gold",
                embedding_vector=[0.1] * 768,
                geological_text="Gold exploration in Pilbara",
                mineral_type="gold",
                coordinates=(120.0, -20.0),
                metadata={"grade": "high", "depth": "shallow"},
                created_at=datetime.now().isoformat(),
                quality_score=0.8
            ),
            VectorRecord(
                id="test_2",
                partition_key="iron",
                embedding_vector=[0.2] * 768,
                geological_text="Iron ore deposits with hematite",
                mineral_type="iron",
                coordinates=(121.0, -21.0),
                metadata={"grade": "medium", "depth": "deep"},
                created_at=datetime.now().isoformat(),
                quality_score=0.7
            )
        ]
        
        # Mock successful storage
        mock_container.create_item.return_value = {}
        
        # Test vector storage
        result = vector_db_manager.store_geological_vectors(test_vectors)
        
        # Verify storage results
        assert result['total_vectors'] == 2
        assert result['successful_inserts'] == 2
        assert result['failed_inserts'] == 0
        assert result['success_rate_percentage'] == 100.0
        assert result['performance_target_met'] == True
    
    def test_vector_query(self, vector_db_manager, mock_cosmos_client):
        """Test vector similarity query"""
        mock_client, mock_container = mock_cosmos_client
        
        # Mock query results
        mock_query_results = [
            {
                'id': 'test_1',
                'partitionKey': 'gold',
                'embedding_vector': [0.1] * 768,
                'geological_text': 'Gold exploration in Pilbara',
                'mineral_type': 'gold',
                'coordinates': {'longitude': 120.0, 'latitude': -20.0},
                'metadata': {'grade': 'high'},
                'created_at': datetime.now().isoformat(),
                'quality_score': 0.8
            }
        ]
        
        mock_container.query_items.return_value = mock_query_results
        
        # Test vector query
        query_vector = [0.1] * 768
        results = vector_db_manager.query_similar_vectors(query_vector, top_k=10)
        
        # Verify query results
        assert len(results) == 1
        assert isinstance(results[0], VectorRecord)
        assert results[0].id == 'test_1'
        assert results[0].mineral_type == 'gold'
        assert results[0].quality_score == 0.8
    
    def test_batch_geological_query(self, vector_db_manager, mock_cosmos_client):
        """Test batch query by geological terms"""
        mock_client, mock_container = mock_cosmos_client
        
        # Mock query results
        mock_query_results = [
            {
                'id': 'test_1',
                'partitionKey': 'gold',
                'embedding_vector': [0.1] * 768,
                'geological_text': 'Gold exploration in Pilbara',
                'mineral_type': 'gold',
                'coordinates': {'longitude': 120.0, 'latitude': -20.0},
                'metadata': {'grade': 'high'},
                'created_at': datetime.now().isoformat(),
                'quality_score': 0.8
            }
        ]
        
        mock_container.query_items.return_value = mock_query_results
        
        # Test batch query
        geological_terms = ['gold', 'exploration', 'deposit']
        results = vector_db_manager.batch_query_by_geological_terms(geological_terms, top_k=20)
        
        # Verify batch query results
        assert len(results) == 1
        assert isinstance(results[0], VectorRecord)
        assert 'gold' in results[0].geological_text.lower()
    
    def test_database_statistics(self, vector_db_manager, mock_cosmos_client):
        """Test database statistics generation"""
        mock_client, mock_container = mock_cosmos_client
        
        # Mock container properties
        mock_container.read.return_value = {'_rid': 1024 * 1024}  # 1MB
        
        # Mock query results for statistics
        mock_count_result = [100]  # 100 total records
        mock_mineral_results = [
            {'mineral_type': 'gold', 'count': 50},
            {'mineral_type': 'iron', 'count': 30},
            {'mineral_type': 'copper', 'count': 20}
        ]
        mock_quality_results = [
            {'quality_category': 'excellent', 'count': 20},
            {'quality_category': 'good', 'count': 40},
            {'quality_category': 'fair', 'count': 30},
            {'quality_category': 'poor', 'count': 10}
        ]
        
        # Mock different query calls
        def mock_query_items(query, **kwargs):
            if 'COUNT(1)' in query:
                return mock_count_result
            elif 'mineral_type' in query:
                return mock_mineral_results
            elif 'quality_category' in query:
                return mock_quality_results
            else:
                return []
        
        mock_container.query_items.side_effect = mock_query_items
        
        # Test statistics generation
        stats = vector_db_manager.get_database_statistics()
        
        # Verify statistics
        assert stats['total_records'] == 100
        assert stats['storage_usage_mb'] > 0
        assert 'mineral_type_distribution' in stats
        assert 'quality_score_distribution' in stats
        assert 'performance_metrics' in stats
        assert stats['database_health']['connection_status'] == 'healthy'

class TestGeologicalSemanticSearch:
    """Test semantic search functionality"""
    
    @pytest.fixture
    def search_engine(self):
        """Create semantic search engine"""
        return GeologicalSemanticSearch(vector_dimension=768)
    
    def test_vector_index_initialization(self, search_engine):
        """Test FAISS vector index initialization"""
        assert search_engine.faiss_index is not None
        assert search_engine.vector_dimension == 768
    
    def test_embedding_addition(self, search_engine):
        """Test adding geological embeddings to search index"""
        test_embeddings = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ]
        
        test_metadata = [
            {
                'record_id': 'test_1',
                'description': 'Gold exploration in Pilbara',
                'mineral_type': 'gold',
                'longitude': 120.0,
                'latitude': -20.0
            },
            {
                'record_id': 'test_2',
                'description': 'Iron ore deposits',
                'mineral_type': 'iron',
                'longitude': 121.0,
                'latitude': -21.0
            },
            {
                'record_id': 'test_3',
                'description': 'Copper mineralization',
                'mineral_type': 'copper',
                'longitude': 122.0,
                'latitude': -22.0
            }
        ]
        
        # Test adding embeddings
        success = search_engine.add_geological_embeddings(test_embeddings, test_metadata)
        
        # Verify addition
        assert success == True
        assert search_engine.faiss_index.ntotal == 3
        assert len(search_engine.geological_metadata) == 3
    
    def test_similarity_search(self, search_engine):
        """Test similarity search functionality"""
        # Add test embeddings first
        test_embeddings = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ]
        
        test_metadata = [
            {
                'record_id': 'test_1',
                'description': 'Gold exploration in Pilbara',
                'mineral_type': 'gold',
                'longitude': 120.0,
                'latitude': -20.0
            },
            {
                'record_id': 'test_2',
                'description': 'Iron ore deposits',
                'mineral_type': 'iron',
                'longitude': 121.0,
                'latitude': -21.0
            },
            {
                'record_id': 'test_3',
                'description': 'Copper mineralization',
                'mineral_type': 'copper',
                'longitude': 122.0,
                'latitude': -22.0
            }
        ]
        
        search_engine.add_geological_embeddings(test_embeddings, test_metadata)
        
        # Test similarity search
        query_embedding = [0.15] * 768  # Similar to first embedding
        results = search_engine.search_similar_geological_content(query_embedding, top_k=2)
        
        # Verify search results
        assert len(results) == 2
        assert all(isinstance(result, SimilarityMatch) for result in results)
        assert all(result.similarity_score > 0 for result in results)
        assert results[0].relevance_rank == 1
        assert results[1].relevance_rank == 2
    
    def test_geological_query_search(self, search_engine):
        """Test end-to-end geological query search"""
        # Mock Cortex client
        mock_cortex_client = Mock()
        mock_embedding_result = Mock()
        mock_embedding_result.success = True
        mock_embedding_result.embedding_vector = [0.1] * 768
        mock_cortex_client.generate_embeddings_batch.return_value = [mock_embedding_result]
        
        # Add test embeddings
        test_embeddings = [
            [0.1] * 768,
            [0.2] * 768
        ]
        
        test_metadata = [
            {
                'record_id': 'test_1',
                'description': 'Gold exploration in Pilbara',
                'mineral_type': 'gold',
                'longitude': 120.0,
                'latitude': -20.0
            },
            {
                'record_id': 'test_2',
                'description': 'Iron ore deposits',
                'mineral_type': 'iron',
                'longitude': 121.0,
                'latitude': -21.0
            }
        ]
        
        search_engine.add_geological_embeddings(test_embeddings, test_metadata)
        
        # Test geological query search
        query_text = "Gold exploration in Western Australia"
        results = search_engine.search_by_geological_query(query_text, mock_cortex_client, top_k=5)
        
        # Verify search results
        assert len(results) > 0
        assert all(isinstance(result, SimilarityMatch) for result in results)
    
    def test_search_quality_evaluation(self, search_engine):
        """Test search quality evaluation"""
        # Add test embeddings
        test_embeddings = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ]
        
        test_metadata = [
            {
                'record_id': 'test_1',
                'description': 'Gold exploration in Pilbara',
                'mineral_type': 'gold',
                'longitude': 120.0,
                'latitude': -20.0
            },
            {
                'record_id': 'test_2',
                'description': 'Iron ore deposits',
                'mineral_type': 'iron',
                'longitude': 121.0,
                'latitude': -21.0
            },
            {
                'record_id': 'test_3',
                'description': 'Copper mineralization',
                'mineral_type': 'copper',
                'longitude': 122.0,
                'latitude': -22.0
            }
        ]
        
        search_engine.add_geological_embeddings(test_embeddings, test_metadata)
        
        # Test search quality evaluation
        test_queries = [
            {
                'query': 'gold exploration',
                'expected_minerals': ['gold'],
                'expected_locations': ['pilbara']
            },
            {
                'query': 'iron ore',
                'expected_minerals': ['iron'],
                'expected_locations': []
            }
        ]
        
        evaluation_result = search_engine.evaluate_search_quality(test_queries)
        
        # Verify evaluation results
        assert evaluation_result['total_test_queries'] == 2
        assert evaluation_result['accurate_results'] >= 0
        assert evaluation_result['accuracy_percentage'] >= 0
        assert evaluation_result['average_search_time_ms'] > 0
        assert 'performance_target_met' in evaluation_result
    
    def test_search_performance_report(self, search_engine):
        """Test search performance report generation"""
        # Add some test data
        test_embeddings = [[0.1] * 768]
        test_metadata = [{
            'record_id': 'test_1',
            'description': 'Gold exploration',
            'mineral_type': 'gold',
            'longitude': 120.0,
            'latitude': -20.0
        }]
        
        search_engine.add_geological_embeddings(test_embeddings, test_metadata)
        
        # Generate performance report
        report = search_engine.get_search_performance_report()
        
        # Verify report structure
        assert 'search_performance_metrics' in report
        assert 'index_statistics' in report
        assert 'learning_targets_assessment' in report
        assert 'optimization_recommendations' in report
        
        # Verify performance metrics
        metrics = report['search_performance_metrics']
        assert metrics['query_count'] >= 0
        assert metrics['average_search_time_ms'] >= 0
        assert metrics['total_vectors_searched'] >= 0

class TestVectorOperationsPerformance:
    """Test vector operations performance"""
    
    def test_large_scale_vector_storage(self, vector_db_manager, mock_cosmos_client):
        """Test large-scale vector storage performance"""
        mock_client, mock_container = mock_cosmos_client
        
        # Create large test dataset
        large_vectors = []
        for i in range(1000):
            vector = VectorRecord(
                id=f"test_{i}",
                partition_key="gold" if i % 2 == 0 else "iron",
                embedding_vector=[0.1] * 768,
                geological_text=f"Geological description {i}",
                mineral_type="gold" if i % 2 == 0 else "iron",
                coordinates=(120.0 + i/1000, -20.0 + i/1000),
                metadata={"index": i},
                created_at=datetime.now().isoformat(),
                quality_score=0.7 + (i % 3) * 0.1
            )
            large_vectors.append(vector)
        
        # Mock successful storage
        mock_container.create_item.return_value = {}
        
        # Test large-scale storage
        result = vector_db_manager.store_geological_vectors(large_vectors)
        
        # Verify performance targets
        assert result['total_vectors'] == 1000
        assert result['success_rate_percentage'] >= 95
        assert result['performance_target_met'] == True
    
    def test_high_performance_similarity_search(self, search_engine):
        """Test high-performance similarity search"""
        # Add large number of embeddings
        large_embeddings = []
        large_metadata = []
        
        for i in range(1000):
            large_embeddings.append([0.1 + i/10000] * 768)
            large_metadata.append({
                'record_id': f'test_{i}',
                'description': f'Geological description {i}',
                'mineral_type': 'gold' if i % 2 == 0 else 'iron',
                'longitude': 120.0 + i/1000,
                'latitude': -20.0 + i/1000
            })
        
        search_engine.add_geological_embeddings(large_embeddings, large_metadata)
        
        # Test high-performance search
        query_embedding = [0.15] * 768
        start_time = time.time()
        results = search_engine.search_similar_geological_content(query_embedding, top_k=10)
        search_time = (time.time() - start_time) * 1000
        
        # Verify performance targets
        assert len(results) == 10
        assert search_time < 100  # <100ms target
        assert all(result.similarity_score > 0 for result in results)

if __name__ == '__main__':
    pytest.main([__file__])
