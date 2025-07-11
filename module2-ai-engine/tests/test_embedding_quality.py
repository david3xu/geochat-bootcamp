"""
AI Quality Validation Tests
Validation of geological embedding quality and relevance
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.embedding_processor import GeologicalTextProcessor, EmbeddingProcessor, ProcessedText
from src.snowflake_cortex_client import SnowflakeCortexClient

class TestGeologicalTextProcessor:
    """Test geological text processing quality"""
    
    @pytest.fixture
    def text_processor(self):
        """Create text processor for testing"""
        return GeologicalTextProcessor()
    
    def test_geological_text_processing(self, text_processor):
        """Test geological text processing quality"""
        test_text = "Gold exploration in Pilbara region shows high-grade hematite deposits with 60% Fe content"
        
        processed = text_processor.process_geological_text(test_text)
        
        # Verify processing results
        assert isinstance(processed, ProcessedText)
        assert processed.original_text == test_text
        assert len(processed.cleaned_text) > 0
        assert len(processed.tokens) > 0
        assert len(processed.geological_terms) > 0
        assert len(processed.mineral_types) > 0
        assert processed.quality_score > 0
        assert processed.processing_time_ms > 0
    
    def test_geological_term_extraction(self, text_processor):
        """Test geological terminology extraction"""
        test_texts = [
            "Gold mineralization in quartz veins",
            "Iron ore deposits with magnetite",
            "Copper exploration drilling results"
        ]
        
        for text in test_texts:
            processed = text_processor.process_geological_text(text)
            
            # Verify geological terms are extracted
            assert len(processed.geological_terms) > 0
            assert any(term in processed.geological_terms for term in ['mineral', 'ore', 'deposit', 'exploration'])
    
    def test_mineral_type_extraction(self, text_processor):
        """Test mineral type identification"""
        test_texts = [
            "Gold deposits in Western Australia",
            "Iron ore mining operations",
            "Copper and nickel exploration"
        ]
        
        for text in test_texts:
            processed = text_processor.process_geological_text(text)
            
            # Verify mineral types are identified
            assert len(processed.mineral_types) > 0
            assert any(mineral in processed.mineral_types for mineral in ['gold', 'iron', 'copper', 'nickel'])
    
    def test_quality_score_calculation(self, text_processor):
        """Test quality score calculation for geological texts"""
        high_quality_text = "Gold exploration in Pilbara region reveals high-grade hematite deposits with 60% Fe content and extensive mineralization"
        low_quality_text = "Some rocks here"
        
        high_quality_result = text_processor.process_geological_text(high_quality_text)
        low_quality_result = text_processor.process_geological_text(low_quality_text)
        
        # Verify quality scoring
        assert high_quality_result.quality_score > low_quality_result.quality_score
        assert high_quality_result.quality_score >= 0.6  # High quality threshold
        assert low_quality_result.quality_score < 0.4    # Low quality threshold
    
    def test_batch_text_processing(self, text_processor):
        """Test batch processing of geological texts"""
        test_texts = [
            "Gold exploration in Pilbara",
            "Iron ore deposits with hematite",
            "Copper mineralization in volcanic rocks",
            "Nickel laterite exploration"
        ]
        
        processed_texts = text_processor.batch_process_texts(test_texts)
        
        # Verify batch processing
        assert len(processed_texts) == len(test_texts)
        assert all(isinstance(pt, ProcessedText) for pt in processed_texts)
        assert all(pt.quality_score > 0 for pt in processed_texts)

class TestEmbeddingProcessor:
    """Test embedding generation and quality management"""
    
    @pytest.fixture
    def mock_cortex_client(self):
        """Create mock Cortex client"""
        mock_client = Mock(spec=SnowflakeCortexClient)
        return mock_client
    
    @pytest.fixture
    def embedding_processor(self, mock_cortex_client):
        """Create embedding processor with mock client"""
        return EmbeddingProcessor(mock_cortex_client)
    
    def test_geological_embedding_generation(self, embedding_processor, mock_cortex_client):
        """Test geological embedding generation with quality filtering"""
        test_texts = [
            "Gold exploration in Pilbara region",
            "Iron ore deposits with high grade",
            "Copper mineralization in volcanic rocks"
        ]
        
        # Mock embedding results
        mock_embeddings = [
            Mock(success=True, embedding_vector=[0.1] * 768, processing_time_ms=100),
            Mock(success=True, embedding_vector=[0.1] * 768, processing_time_ms=120),
            Mock(success=True, embedding_vector=[0.1] * 768, processing_time_ms=110)
        ]
        mock_cortex_client.generate_embeddings_batch.return_value = mock_embeddings
        
        # Test embedding generation
        result = embedding_processor.generate_geological_embeddings(test_texts)
        
        # Verify results
        assert result.total_texts == 3
        assert result.successful_embeddings == 3
        assert result.failed_embeddings == 0
        assert result.average_quality_score > 0
        assert result.processing_time_seconds > 0
        assert len(result.embeddings) == 3
    
    def test_quality_threshold_filtering(self, embedding_processor, mock_cortex_client):
        """Test quality threshold filtering for embeddings"""
        high_quality_texts = [
            "Gold exploration in Pilbara region with high-grade hematite deposits",
            "Iron ore mining operations with extensive mineralization"
        ]
        
        low_quality_texts = [
            "Some rocks here",
            "Basic text"
        ]
        
        all_texts = high_quality_texts + low_quality_texts
        
        # Mock embedding results
        mock_embeddings = [
            Mock(success=True, embedding_vector=[0.1] * 768, processing_time_ms=100)
            for _ in range(len(high_quality_texts))
        ]
        mock_cortex_client.generate_embeddings_batch.return_value = mock_embeddings
        
        # Test with quality threshold
        result = embedding_processor.generate_geological_embeddings(all_texts)
        
        # Verify quality filtering
        assert result.total_texts == len(all_texts)
        assert result.successful_embeddings <= len(high_quality_texts)  # Only high quality texts
        assert result.average_quality_score >= 0.3  # Quality threshold
    
    def test_large_dataset_processing(self, embedding_processor, mock_cortex_client):
        """Test large dataset processing performance"""
        large_dataset = [f"Geological description {i} with gold exploration" for i in range(1000)]
        
        # Mock batch processing results
        mock_processing_report = {
            'total_texts_processed': 1000,
            'successful_embeddings': 950,
            'failed_embeddings': 50,
            'success_rate_percentage': 95.0,
            'total_processing_time_seconds': 300,
            'processing_rate_per_second': 3.33,
            'performance_target_met': True
        }
        mock_cortex_client.batch_process_large_dataset.return_value = mock_processing_report
        
        # Test large dataset processing
        result = embedding_processor.process_large_geological_dataset(large_dataset)
        
        # Verify processing results
        assert result['total_texts_processed'] == 1000
        assert result['successful_embeddings'] == 950
        assert result['success_rate_percentage'] >= 95.0
        assert result['performance_target_met'] == True
        assert 'quality_distribution' in result
    
    def test_embedding_quality_evaluation(self, embedding_processor):
        """Test embedding quality evaluation with test cases"""
        test_texts = [
            "Gold exploration in Pilbara region with high-grade deposits",
            "Iron ore mining with hematite and magnetite",
            "Copper mineralization in volcanic rocks",
            "Nickel laterite exploration in Western Australia"
        ]
        
        evaluation_result = embedding_processor.evaluate_embedding_quality(test_texts)
        
        # Verify evaluation results
        assert evaluation_result['total_test_texts'] == len(test_texts)
        assert len(evaluation_result['quality_scores']) == len(test_texts)
        assert len(evaluation_result['geological_term_counts']) == len(test_texts)
        assert len(evaluation_result['mineral_type_counts']) == len(test_texts)
        
        # Verify summary statistics
        summary = evaluation_result['summary_statistics']
        assert summary['average_quality_score'] > 0
        assert summary['average_geological_terms'] > 0
        assert summary['average_mineral_types'] > 0
        assert summary['average_processing_time_ms'] > 0

class TestQualityMetrics:
    """Test quality metrics and assessment"""
    
    def test_quality_distribution_calculation(self, embedding_processor):
        """Test quality score distribution calculation"""
        # Create test processed texts with different quality scores
        from src.embedding_processor import ProcessedText
        
        test_processed_texts = [
            ProcessedText("excellent text", "excellent text", [], [], [], 0.0, 0.9),
            ProcessedText("good text", "good text", [], [], [], 0.0, 0.7),
            ProcessedText("fair text", "fair text", [], [], [], 0.0, 0.5),
            ProcessedText("poor text", "poor text", [], [], [], 0.0, 0.3),
            ProcessedText("very poor text", "very poor text", [], [], [], 0.0, 0.1)
        ]
        
        distribution = embedding_processor._calculate_quality_distribution(test_processed_texts)
        
        # Verify distribution calculation
        assert distribution['excellent'] == 1
        assert distribution['good'] == 1
        assert distribution['fair'] == 1
        assert distribution['poor'] == 1
        assert distribution['very_poor'] == 1
        assert sum(distribution.values()) == len(test_processed_texts)
    
    def test_geological_term_extraction_accuracy(self, text_processor):
        """Test accuracy of geological term extraction"""
        test_cases = [
            ("Gold exploration in Pilbara", ['gold', 'exploration']),
            ("Iron ore deposits with hematite", ['iron', 'ore', 'deposit', 'hematite']),
            ("Copper mineralization in volcanic rocks", ['copper', 'mineralization', 'volcanic'])
        ]
        
        for text, expected_terms in test_cases:
            processed = text_processor.process_geological_text(text)
            
            # Check that expected terms are extracted
            extracted_terms = [term.lower() for term in processed.geological_terms]
            for expected_term in expected_terms:
                assert any(expected_term in term for term in extracted_terms), f"Expected term '{expected_term}' not found in extracted terms: {extracted_terms}"
    
    def test_mineral_type_identification_accuracy(self, text_processor):
        """Test accuracy of mineral type identification"""
        test_cases = [
            ("Gold deposits in Western Australia", ['gold']),
            ("Iron ore and copper exploration", ['iron', 'copper']),
            ("Nickel laterite and lithium brine", ['nickel', 'lithium'])
        ]
        
        for text, expected_minerals in test_cases:
            processed = text_processor.process_geological_text(text)
            
            # Check that expected minerals are identified
            extracted_minerals = [mineral.lower() for mineral in processed.mineral_types]
            for expected_mineral in expected_minerals:
                assert expected_mineral in extracted_minerals, f"Expected mineral '{expected_mineral}' not found in extracted minerals: {extracted_minerals}"

if __name__ == '__main__':
    pytest.main([__file__])
