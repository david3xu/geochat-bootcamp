"""
Module 2: AI Engine with Snowflake Cortex
Main application entry point for geological AI system
"""
import logging
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from snowflake_cortex_client import SnowflakeCortexClient
from embedding_processor import EmbeddingProcessor
from semantic_search import GeologicalSemanticSearch
from qa_engine import GeologicalQAEngine
from vector_database import VectorDatabaseManager
from performance_monitor import AIPerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/module2_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def initialize_ai_system():
    """Initialize the complete AI system for Module 2"""
    logger.info("🚀 Initializing Module 2: AI Engine with Snowflake Cortex")
    
    try:
        # Initialize Snowflake Cortex client
        logger.info("📡 Connecting to Snowflake Cortex...")
        cortex_client = SnowflakeCortexClient()
        
        # Initialize embedding processor
        logger.info("🔧 Initializing embedding processor...")
        embedding_processor = EmbeddingProcessor(cortex_client)
        
        # Initialize semantic search engine
        logger.info("🔍 Initializing semantic search engine...")
        search_engine = GeologicalSemanticSearch(vector_dimension=768)
        
        # Initialize vector database
        logger.info("💾 Initializing vector database...")
        vector_db = VectorDatabaseManager()
        
        # Initialize QA engine
        logger.info("❓ Initializing geological QA engine...")
        qa_engine = GeologicalQAEngine(cortex_client, search_engine)
        
        # Initialize performance monitor
        logger.info("📊 Initializing performance monitor...")
        performance_monitor = AIPerformanceMonitor()
        
        logger.info("✅ AI system initialization completed successfully")
        
        return {
            'cortex_client': cortex_client,
            'embedding_processor': embedding_processor,
            'search_engine': search_engine,
            'vector_db': vector_db,
            'qa_engine': qa_engine,
            'performance_monitor': performance_monitor
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI system: {str(e)}")
        raise

def run_geological_qa_demo(ai_system):
    """Run geological QA demonstration"""
    logger.info("🎯 Running geological QA demonstration")
    
    qa_engine = ai_system['qa_engine']
    
    # Sample geological questions
    test_questions = [
        "What are the main gold exploration areas in Western Australia?",
        "How does iron ore formation occur in the Pilbara region?",
        "What geological processes create copper porphyry deposits?",
        "Explain the difference between laterite and sulfide nickel deposits",
        "What are the key exploration techniques for lithium pegmatites?"
    ]
    
    for question in test_questions:
        logger.info(f"Question: {question}")
        try:
            response = qa_engine.process_geological_query(question)
            logger.info(f"Answer: {response.answer[:200]}...")
            logger.info(f"Confidence: {response.confidence_score:.2f}")
            logger.info(f"Geological Accuracy: {response.geological_accuracy:.2f}")
            logger.info(f"Processing Time: {response.processing_time_ms:.2f}ms")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")

def run_embedding_generation_demo(ai_system):
    """Run embedding generation demonstration"""
    logger.info("🔤 Running embedding generation demonstration")
    
    embedding_processor = ai_system['embedding_processor']
    
    # Sample geological texts
    sample_texts = [
        "Gold exploration in Pilbara region reveals high-grade hematite deposits",
        "Iron ore mining operations with extensive mineralization",
        "Copper porphyry deposits in volcanic arc settings",
        "Nickel laterite exploration in tropical weathering profiles",
        "Lithium pegmatite deposits with rare earth elements"
    ]
    
    try:
        # Generate embeddings
        result = embedding_processor.generate_geological_embeddings(sample_texts)
        
        logger.info(f"Embedding Generation Results:")
        logger.info(f"  Total texts: {result.total_texts}")
        logger.info(f"  Successful embeddings: {result.successful_embeddings}")
        logger.info(f"  Failed embeddings: {result.failed_embeddings}")
        logger.info(f"  Average quality score: {result.average_quality_score:.2f}")
        logger.info(f"  Processing time: {result.processing_time_seconds:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}")

def run_vector_search_demo(ai_system):
    """Run vector search demonstration"""
    logger.info("🔍 Running vector search demonstration")
    
    search_engine = ai_system['search_engine']
    cortex_client = ai_system['cortex_client']
    
    # Add sample embeddings to search index
    sample_embeddings = [
        [0.1] * 768,
        [0.2] * 768,
        [0.3] * 768
    ]
    
    sample_metadata = [
        {
            'record_id': 'GOLD_001',
            'description': 'Gold exploration in Pilbara',
            'mineral_type': 'gold',
            'longitude': 120.1234,
            'latitude': -20.5678
        },
        {
            'record_id': 'IRON_001',
            'description': 'Iron ore deposits',
            'mineral_type': 'iron',
            'longitude': 120.3456,
            'latitude': -20.7890
        },
        {
            'record_id': 'COPPER_001',
            'description': 'Copper mineralization',
            'mineral_type': 'copper',
            'longitude': 121.5678,
            'latitude': -21.9012
        }
    ]
    
    try:
        # Add embeddings to search index
        search_engine.add_geological_embeddings(sample_embeddings, sample_metadata)
        
        # Test similarity search
        query_text = "Gold exploration in Western Australia"
        results = search_engine.search_by_geological_query(query_text, cortex_client, top_k=3)
        
        logger.info(f"Vector Search Results for '{query_text}':")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result.geological_text[:100]}...")
            logger.info(f"     Similarity: {result.similarity_score:.3f}")
            logger.info(f"     Mineral: {result.mineral_type}")
        
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")

def run_performance_monitoring_demo(ai_system):
    """Run performance monitoring demonstration"""
    logger.info("📊 Running performance monitoring demonstration")
    
    performance_monitor = ai_system['performance_monitor']
    
    try:
        # Start monitoring
        performance_monitor.start_monitoring(interval_seconds=10)
        
        # Simulate some activity
        time.sleep(5)
        
        # Get performance report
        report = performance_monitor.get_current_performance_report()
        
        logger.info("Performance Report:")
        logger.info(f"  Cortex embed calls: {report['current_metrics']['cortex_embed_calls']}")
        logger.info(f"  Cortex complete calls: {report['current_metrics']['cortex_complete_calls']}")
        logger.info(f"  Average embed time: {report['current_metrics']['average_embed_time_ms']:.2f}ms")
        logger.info(f"  Average complete time: {report['current_metrics']['average_complete_time_ms']:.2f}ms")
        logger.info(f"  Performance compliance: {report['current_metrics']['performance_target_compliance']:.1f}%")
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Error in performance monitoring: {str(e)}")

def generate_learning_report(ai_system):
    """Generate comprehensive learning report for Module 2"""
    logger.info("📋 Generating Module 2 learning report")
    
    try:
        cortex_client = ai_system['cortex_client']
        qa_engine = ai_system['qa_engine']
        performance_monitor = ai_system['performance_monitor']
        
        # Get Cortex usage report
        daily_report = cortex_client.generate_daily_usage_report()
        
        # Get QA performance report
        qa_report = qa_engine.get_qa_performance_report()
        
        # Get performance monitoring report
        perf_report = performance_monitor.get_current_performance_report()
        
        # Compile comprehensive report
        learning_report = {
            'module': 'Module 2: AI Engine with Snowflake Cortex',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cortex_usage_summary': daily_report['cortex_usage_summary'],
            'learning_targets_assessment': daily_report['learning_targets_assessment'],
            'qa_performance': qa_report['learning_targets_assessment'],
            'performance_metrics': perf_report['learning_targets_assessment'],
            'supervision_metrics': {
                'cortex_calls_total': daily_report['cortex_usage_summary']['total_calls'],
                'qa_accuracy_achieved': qa_report['learning_targets_assessment']['accuracy_target_85_percent'],
                'performance_targets_met': perf_report['learning_targets_assessment']['performance_compliance_percentage']
            }
        }
        
        # Save report
        import json
        with open('reports/module2_learning_report.json', 'w') as f:
            json.dump(learning_report, f, indent=2)
        
        logger.info("✅ Learning report generated successfully")
        logger.info(f"📄 Report saved to: reports/module2_learning_report.json")
        
        return learning_report
        
    except Exception as e:
        logger.error(f"Error generating learning report: {str(e)}")
        return None

def main():
    """Main application entry point"""
    logger.info("🎓 Module 2: AI Engine with Snowflake Cortex")
    logger.info("Full Stack AI Engineer Bootcamp - Week 2 Implementation")
    logger.info("=" * 80)
    
    try:
        # Initialize AI system
        ai_system = initialize_ai_system()
        
        # Run demonstrations
        run_embedding_generation_demo(ai_system)
        run_vector_search_demo(ai_system)
        run_geological_qa_demo(ai_system)
        run_performance_monitoring_demo(ai_system)
        
        # Generate learning report
        learning_report = generate_learning_report(ai_system)
        
        if learning_report:
            logger.info("🎉 Module 2 demonstration completed successfully!")
            logger.info("📊 Learning outcomes achieved:")
            logger.info(f"  - Cortex calls: {learning_report['supervision_metrics']['cortex_calls_total']}")
            logger.info(f"  - QA accuracy: {learning_report['qa_performance']['accuracy_target_85_percent']}")
            logger.info(f"  - Performance compliance: {learning_report['performance_metrics']['performance_compliance_percentage']}%")
        
    except Exception as e:
        logger.error(f"❌ Module 2 execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 