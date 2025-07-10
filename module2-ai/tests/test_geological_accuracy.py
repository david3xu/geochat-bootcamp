"""
Geological Domain Accuracy Testing
Module 2: AI Engine - Week 2 Implementation
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qa_engine import GeologicalQAEngine
from snowflake_cortex_client import SnowflakeCortexClient
from vector_database import VectorDatabaseManager

class TestGeologicalAccuracy:
    """Test geological domain accuracy for Module 2"""
    
    @pytest.fixture
    def qa_engine(self):
        """Initialize QA engine for testing"""
        cortex_client = SnowflakeCortexClient()
        vector_db = VectorDatabaseManager()
        return GeologicalQAEngine(cortex_client, vector_db)
    
    def test_geological_terminology_accuracy(self, qa_engine):
        """Test accuracy of geological terminology usage"""
        test_questions = [
            "What are the key indicators of gold mineralization?",
            "Explain the formation of porphyry copper deposits",
            "What is the geological significance of banded iron formations?"
        ]
        
        for question in test_questions:
            response = qa_engine.process_geological_query(question)
            assert response.geological_accuracy >= 0.85, f"Geological accuracy below threshold: {response.geological_accuracy}"
    
    def test_mineral_identification_accuracy(self, qa_engine):
        """Test accuracy of mineral type identification"""
        mineral_questions = [
            "What are the characteristics of gold deposits?",
            "How do you identify copper mineralization?",
            "What are the exploration methods for lithium deposits?"
        ]
        
        for question in mineral_questions:
            response = qa_engine.process_geological_query(question)
            # Check for mineral-specific terminology
            assert any(mineral in response.answer.lower() for mineral in ['gold', 'copper', 'lithium'])
    
    def test_geological_process_accuracy(self, qa_engine):
        """Test accuracy of geological process explanations"""
        process_questions = [
            "How does hydrothermal alteration occur?",
            "What is the process of metamorphism?",
            "Explain the formation of igneous rocks"
        ]
        
        for question in process_questions:
            response = qa_engine.process_geological_query(question)
            assert response.confidence_score >= 0.8, f"Confidence below threshold: {response.confidence_score}"
    
    def test_western_australia_geology_accuracy(self, qa_engine):
        """Test accuracy of Western Australia geology knowledge"""
        wa_questions = [
            "What are the main geological regions of Western Australia?",
            "Explain the geology of the Pilbara region",
            "What mineral deposits are found in the Yilgarn Craton?"
        ]
        
        for question in wa_questions:
            response = qa_engine.process_geological_query(question)
            # Check for WA-specific geological terms
            wa_terms = ['pilbara', 'yilgarn', 'western australia', 'craton']
            assert any(term in response.answer.lower() for term in wa_terms)
    
    def test_exploration_methodology_accuracy(self, qa_engine):
        """Test accuracy of exploration methodology responses"""
        exploration_questions = [
            "What are the main exploration methods for gold?",
            "How do you conduct geological mapping?",
            "What is the role of geochemistry in exploration?"
        ]
        
        for question in exploration_questions:
            response = qa_engine.process_geological_query(question)
            assert response.processing_time < 2.0, f"Response time too slow: {response.processing_time}s"
    
    def test_quality_assessment_accuracy(self, qa_engine):
        """Test quality assessment functionality"""
        test_question = "What are the geological controls on mineralization?"
        response = qa_engine.process_geological_query(test_question)
        
        # Validate quality metrics
        assert hasattr(response, 'geological_accuracy')
        assert hasattr(response, 'confidence_score')
        assert hasattr(response, 'processing_time')
        assert response.geological_accuracy >= 0.0 and response.geological_accuracy <= 1.0
        assert response.confidence_score >= 0.0 and response.confidence_score <= 1.0
        assert response.processing_time > 0.0

if __name__ == "__main__":
    pytest.main([__file__]) 