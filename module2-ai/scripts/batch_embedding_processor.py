#!/usr/bin/env python3
"""
Batch Embedding Processor for Geological Data
Module 2: AI Engine - Week 2 Implementation
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from snowflake_cortex_client import SnowflakeCortexClient
from embedding_processor import GeologicalEmbeddingProcessor
from vector_database import VectorDatabaseManager, VectorRecord

def process_batch_embeddings(texts: List[str], batch_size: int = 100) -> Dict[str, Any]:
    """Process large batches of geological texts for embedding generation"""
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}")
    
    # Initialize components
    cortex_client = SnowflakeCortexClient()
    embedding_processor = GeologicalEmbeddingProcessor(cortex_client)
    vector_db = VectorDatabaseManager()
    
    start_time = time.time()
    processed_count = 0
    failed_count = 0
    
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_start = time.time()
            
            # Process batch
            embeddings = cortex_client.generate_embeddings(batch)
            
            # Create vector records
            vector_records = []
            for embedding_result in embeddings:
                vector_record = VectorRecord(
                    id=f"embedding_{processed_count}",
                    text=embedding_result.text,
                    embedding=embedding_result.embedding,
                    metadata={
                        'source': 'batch_processing',
                        'quality_score': embedding_result.quality_score,
                        'model_used': embedding_result.model_used
                    },
                    geological_terms=embedding_processor.extract_geological_terms(embedding_result.text),
                    mineral_types=embedding_processor.extract_mineral_mentions(embedding_result.text),
                    timestamp=time.time()
                )
                vector_records.append(vector_record)
                processed_count += 1
            
            # Store in vector database
            storage_result = vector_db.store_geological_embeddings(vector_records)
            
            batch_time = time.time() - batch_start
            print(f"Processed batch {i//batch_size + 1}: {len(batch)} texts in {batch_time:.2f}s")
            
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        failed_count += len(batch)
    
    total_time = time.time() - start_time
    
    result = {
        'total_texts': len(texts),
        'processed_count': processed_count,
        'failed_count': failed_count,
        'success_rate': (processed_count / len(texts)) * 100 if texts else 0,
        'total_time': total_time,
        'embeddings_per_second': processed_count / total_time if total_time > 0 else 0
    }
    
    print(f"Batch processing completed: {processed_count}/{len(texts)} successful in {total_time:.2f}s")
    return result

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Gold mineralization found in quartz veins in the Pilbara region",
        "Copper deposits associated with porphyry intrusions",
        "Iron ore formation in banded iron formations"
    ]
    
    result = process_batch_embeddings(sample_texts)
    print(f"Processing result: {result}")
