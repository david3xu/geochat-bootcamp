"""
QA Engine Validation Tests
Validation of geological question-answering functionality
"""
import pytest
import time
from unittest.mock import Mock, patch

from src.qa_engine import GeologicalQAEngine, QAResponse
from src.snowflake_cortex_client import SnowflakeCortexClient
from src.semantic_search import GeologicalSemanticSearch

class TestGeologicalQAEngine:
    """Test geological QA engine functionality"""
    
    @pytest.fixture
    def mock_cortex_client(self):
        """Create mock Cortex client"""
        mock_client = Mock(spec=SnowflakeCortexClient)
        return mock_client
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine"""
        mock_engine = Mock(spec=GeologicalSemanticSearch)
        return mock_engine
    
    @pytest.fixture
    def qa_engine(self, mock_cortex_client, mock_search_engine):
        """Create QA engine with mock components"""
        return GeologicalQAEngine(mock_cortex_client, mock_search_engine)
    
    def test_qa_engine_initialization(self, qa_engine):
        """Test QA engine initialization"""
        assert qa_engine.cortex_client is not None
        assert qa_engine.search_engine is not None
        assert qa_engine.performance_metrics is not None
        assert len(qa_engine.geological_knowledge_base['mineral_properties']) > 0
    
    def test_geological_query_processing(self, qa_engine, mock_cortex_client, mock_search_engine):
        """Test geological query processing"""
        # Mock embedding generation
        mock_embedding_result = Mock()
        mock_embedding_result.success = True
        mock_embedding_result.embedding_vector = [0.1] * 768
        mock_cortex_client.generate_embeddings_batch.return_value = [mock_embedding_result]
        
        # Mock similarity search results
        mock_similarity_match = Mock()
        mock_similarity_match.record_id = 'test_1'
        mock_similarity_match.similarity_score = 0.85
        mock_similarity_match.geological_text = 'Gold exploration in Pilbara region'
        mock_similarity_match.mineral_type = 'gold'
        mock_similarity_match.coordinates = (120.0, -20.0)
        mock_similarity_match.metadata = {'grade': 'high'}
        mock_similarity_match.relevance_rank = 1
        
        mock_search_engine.search_similar_geological_content.return_value = [mock_similarity_match]
        
        # Mock completion result
        mock_completion_result = Mock()
        mock_completion_result.success = True
        mock_completion_result.completion_output = "Gold exploration in Western Australia is primarily focused in the Pilbara and Yilgarn Craton regions, where Archean greenstone belts host significant gold mineralization."
        mock_completion_result.relevance_score = 0.9
        mock_completion_result.processing_time_ms = 1500
        
        mock_cortex_client.complete_geological_query.return_value = mock_completion_result
        
        # Test query processing
        question = "What are the main gold exploration areas in Western Australia?"
        response = qa_engine.process_geological_query(question)
        
        # Verify response
        assert isinstance(response, QAResponse)
        assert response.question == question
        assert len(response.answer) > 0
        assert response.confidence_score > 0
        assert response.processing_time_ms > 0
        assert response.geological_accuracy > 0
        assert len(response.source_documents) > 0
    
    def test_query_analysis(self, qa_engine):
        """Test geological query analysis"""
        test_queries = [
            ("Where are gold deposits located?", 'spatial'),
            ("How does iron ore formation occur?", 'procedural'),
            ("What is copper porphyry?", 'definitional'),
            ("What are the grades of nickel deposits?", 'quantitative'),
            ("Explain geological exploration", 'general')
        ]
        
        for query, expected_type in test_queries:
            analysis = qa_engine._analyze_geological_query(query)
            
            assert analysis['query_type'] == expected_type
            assert 'mentioned_minerals' in analysis
            assert 'mentioned_regions' in analysis
            assert 'complexity_level' in analysis
    
    def test_geological_accuracy_assessment(self, qa_engine):
        """Test geological accuracy assessment"""
        test_cases = [
            ("Gold exploration in Pilbara region shows high-grade hematite deposits", 0.8),
            ("Some rocks here", 0.2),
            ("Iron ore mining with hematite and magnetite in Western Australia", 0.9)
        ]
        
        for answer, expected_accuracy in test_cases:
            accuracy = qa_engine._assess_geological_accuracy("test question", answer)
            assert accuracy > 0
            assert accuracy <= 1.0
    
    def test_confidence_score_calculation(self, qa_engine):
        """Test confidence score calculation"""
        # Mock completion result
        mock_completion_result = Mock()
        mock_completion_result.relevance_score = 0.8
        mock_completion_result.processing_time_ms = 1500
        
        # Mock similarity matches
        mock_similarity_match = Mock()
        mock_similarity_match.similarity_score = 0.85
        context_docs = [mock_similarity_match]
        
        confidence = qa_engine._calculate_confidence_score(mock_completion_result, context_docs)
        
        assert confidence > 0
        assert confidence <= 1.0
    
    def test_spatial_context_extraction(self, qa_engine):
        """Test spatial context extraction"""
        # Mock similarity matches with coordinates
        mock_docs = []
        for i in range(3):
            mock_doc = Mock()
            mock_doc.coordinates = (120.0 + i, -20.0 + i)
            mock_doc.mineral_type = 'gold'
            mock_docs.append(mock_doc)
        
        spatial_context = qa_engine._extract_spatial_context(mock_docs)
        
        assert spatial_context is not None
        assert 'centroid_coordinates' in spatial_context
        assert 'bounding_box' in spatial_context
        assert 'dominant_minerals' in spatial_context
        assert spatial_context['total_sites'] == 3
    
    def test_qa_quality_evaluation(self, qa_engine, mock_cortex_client, mock_search_engine):
        """Test QA quality evaluation"""
        # Mock components
        mock_embedding_result = Mock()
        mock_embedding_result.success = True
        mock_embedding_result.embedding_vector = [0.1] * 768
        mock_cortex_client.generate_embeddings_batch.return_value = [mock_embedding_result]
        
        mock_completion_result = Mock()
        mock_completion_result.success = True
        mock_completion_result.completion_output = "Gold exploration in Western Australia"
        mock_completion_result.relevance_score = 0.9
        mock_completion_result.processing_time_ms = 1500
        mock_cortex_client.complete_geological_query.return_value = mock_completion_result
        
        mock_search_engine.search_similar_geological_content.return_value = []
        
        # Test evaluation
        test_questions = [
            {
                'question': 'What are gold exploration areas?',
                'expected_concepts': ['gold', 'exploration', 'western australia'],
                'accuracy_threshold': 0.8
            },
            {
                'question': 'How does iron ore form?',
                'expected_concepts': ['iron', 'ore', 'formation'],
                'accuracy_threshold': 0.8
            }
        ]
        
        evaluation_result = qa_engine.evaluate_qa_quality(test_questions)
        
        # Verify evaluation results
        assert evaluation_result['total_questions'] == 2
        assert len(evaluation_result['accuracy_scores']) == 2
        assert len(evaluation_result['response_times']) == 2
        assert 'summary_statistics' in evaluation_result
        assert 'performance_assessment' in evaluation_result
    
    def test_qa_performance_report(self, qa_engine):
        """Test QA performance report generation"""
        report = qa_engine.get_qa_performance_report()
        
        # Verify report structure
        assert 'qa_performance_metrics' in report
        assert 'learning_targets_assessment' in report
        assert 'geological_expertise_indicators' in report
        assert 'quality_improvement_recommendations' in report
        
        # Verify metrics
        metrics = report['qa_performance_metrics']
        assert metrics['total_questions_processed'] >= 0
        assert metrics['average_response_time_ms'] >= 0
        assert metrics['average_geological_accuracy'] >= 0
        assert metrics['average_confidence_score'] >= 0
    
    def test_geological_knowledge_integration(self, qa_engine):
        """Test geological knowledge base integration"""
        knowledge_base = qa_engine.geological_knowledge_base
        
        # Verify mineral properties
        assert 'gold' in knowledge_base['mineral_properties']
        assert 'iron_ore' in knowledge_base['mineral_properties']
        assert 'copper' in knowledge_base['mineral_properties']
        
        # Verify geological formations
        assert 'western_australia' in knowledge_base['geological_formations']
        assert 'pilbara' in knowledge_base['geological_formations']['western_australia']
        
        # Verify exploration techniques
        assert len(knowledge_base['exploration_techniques']) > 0
        assert 'geophysical_surveys' in knowledge_base['exploration_techniques']
    
    def test_query_complexity_assessment(self, qa_engine):
        """Test query complexity assessment"""
        test_cases = [
            ("What is gold?", 'simple'),
            ("How does iron ore formation occur in the Pilbara region?", 'moderate'),
            ("Explain the complex geological processes that create copper porphyry deposits in volcanic arc settings with specific reference to Western Australian geology", 'complex')
        ]
        
        for query, expected_complexity in test_cases:
            complexity = qa_engine._assess_query_complexity(query)
            assert complexity == expected_complexity
    
    def test_context_filtering(self, qa_engine):
        """Test context filtering for geological relevance"""
        # Mock similarity matches
        mock_docs = []
        for i in range(3):
            mock_doc = Mock()
            mock_doc.geological_text = f"Geological description {i} with gold exploration"
            mock_doc.similarity_score = 0.8 - i * 0.1
            mock_doc.mineral_type = 'gold'
            mock_docs.append(mock_doc)
        
        query = "Gold exploration in Western Australia"
        filtered_docs = qa_engine._filter_context_for_geological_relevance(query, mock_docs)
        
        # Verify filtering
        assert len(filtered_docs) == 3
        assert filtered_docs[0].similarity_score >= filtered_docs[1].similarity_score
    
    def test_enhanced_prompt_construction(self, qa_engine):
        """Test enhanced geological prompt construction"""
        # Mock context documents
        mock_doc = Mock()
        mock_doc.coordinates = (120.0, -20.0)
        mock_doc.geological_text = "Gold exploration in Pilbara region"
        context_docs = [mock_doc]
        
        # Mock query analysis
        query_analysis = {
            'query_type': 'spatial',
            'mentioned_minerals': ['gold'],
            'mentioned_regions': ['pilbara'],
            'requires_spatial_context': True,
            'complexity_level': 'moderate'
        }
        
        question = "Where are gold deposits located in Western Australia?"
        enhanced_prompt = qa_engine._construct_enhanced_geological_prompt(question, context_docs, query_analysis)
        
        # Verify enhanced prompt
        assert len(enhanced_prompt) > 0
        assert "geological consultant" in enhanced_prompt.lower()
        assert "western australia" in enhanced_prompt.lower()
        assert question in enhanced_prompt

class TestQAPerformance:
    """Test QA performance under various conditions"""
    
    def test_qa_response_time_performance(self, qa_engine, mock_cortex_client, mock_search_engine):
        """Test QA response time performance"""
        # Mock fast response
        mock_embedding_result = Mock()
        mock_embedding_result.success = True
        mock_embedding_result.embedding_vector = [0.1] * 768
        mock_cortex_client.generate_embeddings_batch.return_value = [mock_embedding_result]
        
        mock_completion_result = Mock()
        mock_completion_result.success = True
        mock_completion_result.completion_output = "Gold exploration in Western Australia"
        mock_completion_result.relevance_score = 0.9
        mock_completion_result.processing_time_ms = 800  # Fast response
        
        mock_cortex_client.complete_geological_query.return_value = mock_completion_result
        mock_search_engine.search_similar_geological_content.return_value = []
        
        # Test performance
        start_time = time.time()
        response = qa_engine.process_geological_query("What is gold exploration?")
        processing_time = (time.time() - start_time) * 1000
        
        # Verify performance targets
        assert processing_time < 2000  # <2s target
        assert response.processing_time_ms < 2000
    
    def test_qa_accuracy_performance(self, qa_engine, mock_cortex_client, mock_search_engine):
        """Test QA accuracy performance"""
        # Mock high-quality response
        mock_embedding_result = Mock()
        mock_embedding_result.success = True
        mock_embedding_result.embedding_vector = [0.1] * 768
        mock_cortex_client.generate_embeddings_batch.return_value = [mock_embedding_result]
        
        high_quality_answer = "Gold exploration in Western Australia is primarily focused in the Pilbara and Yilgarn Craton regions, where Archean greenstone belts host significant gold mineralization through hydrothermal processes during orogenic events."
        
        mock_completion_result = Mock()
        mock_completion_result.success = True
        mock_completion_result.completion_output = high_quality_answer
        mock_completion_result.relevance_score = 0.95
        mock_completion_result.processing_time_ms = 1500
        
        mock_cortex_client.complete_geological_query.return_value = mock_completion_result
        mock_search_engine.search_similar_geological_content.return_value = []
        
        # Test accuracy
        response = qa_engine.process_geological_query("What are gold exploration areas in Western Australia?")
        
        # Verify accuracy targets
        assert response.geological_accuracy >= 0.85  # 85% accuracy target
        assert response.confidence_score >= 0.8      # 80% confidence target

if __name__ == '__main__':
    pytest.main([__file__]) 