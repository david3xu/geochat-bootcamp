#!/usr/bin/env python3
"""
AI Quality Evaluation Script
Module 2: AI Engine - Week 2 Implementation
"""
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qa_engine import GeologicalQAEngine
from snowflake_cortex_client import SnowflakeCortexClient
from vector_database import VectorDatabaseManager
from performance_monitor import PerformanceMonitor

def evaluate_ai_quality(test_questions: List[str]) -> Dict[str, Any]:
    """Evaluate AI quality for geological question answering"""
    
    print(f"Evaluating AI quality with {len(test_questions)} test questions")
    
    # Initialize components
    cortex_client = SnowflakeCortexClient()
    vector_db = VectorDatabaseManager()
    qa_engine = GeologicalQAEngine(cortex_client, vector_db)
    performance_monitor = PerformanceMonitor()
    
    evaluation_results = {
        'total_questions': len(test_questions),
        'successful_responses': 0,
        'failed_responses': 0,
        'average_response_time': 0.0,
        'average_confidence': 0.0,
        'geological_accuracy_scores': [],
        'quality_scores': []
    }
    
    total_response_time = 0.0
    total_confidence = 0.0
    
    for i, question in enumerate(test_questions):
        try:
            print(f"Processing question {i+1}/{len(test_questions)}: {question[:50]}...")
            
            start_time = time.time()
            response = qa_engine.process_geological_query(question)
            response_time = time.time() - start_time
            
            # Track performance
            performance_monitor.track_operation("qa_evaluation", response_time)
            
            # Collect metrics
            evaluation_results['successful_responses'] += 1
            total_response_time += response_time
            total_confidence += response.confidence_score
            
            evaluation_results['geological_accuracy_scores'].append(response.geological_accuracy)
            evaluation_results['quality_scores'].append(response.confidence_score)
            
            print(f"Response time: {response_time:.2f}s, Confidence: {response.confidence_score:.3f}")
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            evaluation_results['failed_responses'] += 1
            performance_monitor.track_operation("qa_evaluation", 0, success=False, error_message=str(e))
    
    # Calculate averages
    if evaluation_results['successful_responses'] > 0:
        evaluation_results['average_response_time'] = total_response_time / evaluation_results['successful_responses']
        evaluation_results['average_confidence'] = total_confidence / evaluation_results['successful_responses']
        evaluation_results['average_geological_accuracy'] = sum(evaluation_results['geological_accuracy_scores']) / len(evaluation_results['geological_accuracy_scores'])
        evaluation_results['average_quality_score'] = sum(evaluation_results['quality_scores']) / len(evaluation_results['quality_scores'])
    
    # Success rate
    evaluation_results['success_rate'] = (evaluation_results['successful_responses'] / evaluation_results['total_questions']) * 100
    
    print(f"Quality evaluation completed:")
    print(f"  Success rate: {evaluation_results['success_rate']:.1f}%")
    print(f"  Average response time: {evaluation_results['average_response_time']:.2f}s")
    print(f"  Average confidence: {evaluation_results['average_confidence']:.3f}")
    print(f"  Average geological accuracy: {evaluation_results.get('average_geological_accuracy', 0):.3f}")
    
    return evaluation_results

def run_quality_evaluation():
    """Run comprehensive quality evaluation"""
    
    # Test questions for geological domain evaluation
    test_questions = [
        "What are the key indicators of gold mineralization in Western Australia?",
        "How do you explore for porphyry copper deposits?",
        "What is the geological significance of banded iron formations?",
        "Explain the formation of lithium deposits in pegmatites",
        "What are the main exploration methods for uranium deposits?",
        "How do you identify hydrothermal alteration zones?",
        "What is the difference between a resource and a reserve?",
        "Explain the geological controls on mineralization",
        "What are the characteristics of epithermal gold deposits?",
        "How do you conduct geological mapping for exploration?"
    ]
    
    results = evaluate_ai_quality(test_questions)
    
    # Generate evaluation report
    report = {
        'evaluation_timestamp': time.time(),
        'module': 'Module 2: AI Engine',
        'target_metrics': {
            'response_time_target': '<2s',
            'accuracy_target': '85%+',
            'success_rate_target': '95%+'
        },
        'actual_results': results,
        'targets_met': {
            'response_time': results['average_response_time'] < 2.0,
            'accuracy': results.get('average_geological_accuracy', 0) >= 0.85,
            'success_rate': results['success_rate'] >= 95.0
        }
    }
    
    print(f"\nQuality Evaluation Report:")
    print(f"  Response time target met: {report['targets_met']['response_time']}")
    print(f"  Accuracy target met: {report['targets_met']['accuracy']}")
    print(f"  Success rate target met: {report['targets_met']['success_rate']}")
    
    return report

if __name__ == "__main__":
    report = run_quality_evaluation()
