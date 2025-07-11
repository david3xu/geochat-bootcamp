"""
Geological Text Processing and Embedding Generation
Measurable Success: 10,000+ geological embeddings processed with 95%+ quality
"""
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .snowflake_cortex_client import SnowflakeCortexClient, EmbeddingResult
from .config import config

logger = logging.getLogger(__name__)

@dataclass
class ProcessedText:
    """Processed geological text with metadata"""
    original_text: str
    cleaned_text: str
    tokens: List[str]
    geological_terms: List[str]
    mineral_types: List[str]
    processing_time_ms: float
    quality_score: float

@dataclass
class EmbeddingBatchResult:
    """Batch embedding processing result"""
    total_texts: int
    successful_embeddings: int
    failed_embeddings: int
    average_quality_score: float
    processing_time_seconds: float
    embeddings: List[EmbeddingResult]

class GeologicalTextProcessor:
    """
    Geological text preprocessing and quality assessment
    Measurable Success: 95%+ text quality for embedding generation
    """
    
    def __init__(self):
        self.geological_terms = {
            'mineral', 'ore', 'deposit', 'exploration', 'geology', 'formation',
            'gold', 'iron', 'copper', 'nickel', 'lithium', 'uranium',
            'mining', 'drilling', 'assay', 'grade', 'tonnage', 'outcrop',
            'metamorphic', 'igneous', 'sedimentary', 'fault', 'vein', 'lode',
            'hematite', 'magnetite', 'chalcopyrite', 'pentlandite', 'spodumene'
        }
        
        self.mineral_types = {
            'gold', 'iron', 'copper', 'nickel', 'lithium', 'uranium', 'zinc',
            'lead', 'silver', 'platinum', 'palladium', 'diamond', 'coal'
        }
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
            self.lemmatizer = None
            self.stop_words = set()
    
    def process_geological_text(self, text: str) -> ProcessedText:
        """
        Process geological text for optimal embedding generation
        Success Metric: 95%+ text quality for geological domain
        """
        start_time = time.time()
        
        try:
            # Step 1: Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Step 2: Tokenize and lemmatize
            tokens = self._tokenize_and_lemmatize(cleaned_text)
            
            # Step 3: Extract geological terms
            geological_terms = self._extract_geological_terms(tokens)
            mineral_types = self._extract_mineral_types(tokens)
            
            # Step 4: Calculate quality score
            quality_score = self._calculate_text_quality(cleaned_text, geological_terms, mineral_types)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessedText(
                original_text=text,
                cleaned_text=cleaned_text,
                tokens=tokens,
                geological_terms=geological_terms,
                mineral_types=mineral_types,
                processing_time_ms=processing_time,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            return ProcessedText(
                original_text=text,
                cleaned_text=text,
                tokens=[],
                geological_terms=[],
                mineral_types=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                quality_score=0.0
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize geological text"""
        # Remove special characters but preserve geological terminology
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Preserve important geological terms
        geological_patterns = [
            r'Fe\d+',  # Iron grades
            r'Au\d+',  # Gold grades
            r'Cu\d+',  # Copper grades
            r'\d+\.\d+%',  # Percentage grades
            r'\d+\.\d+°[NS]',  # Coordinates
            r'\d+\.\d+°[EW]'
        ]
        
        for pattern in geological_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = cleaned.replace(match, match)
        
        return cleaned
    
    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text while preserving geological terms"""
        if not self.lemmatizer:
            return text.lower().split()
        
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = []
        
        for token in tokens:
            # Preserve geological terms without lemmatization
            if token in self.geological_terms or token in self.mineral_types:
                lemmatized_tokens.append(token)
            else:
                # Lemmatize non-geological terms
                lemmatized = self.lemmatizer.lemmatize(token)
                lemmatized_tokens.append(lemmatized)
        
        # Remove stop words but preserve geological terms
        filtered_tokens = [
            token for token in lemmatized_tokens
            if token not in self.stop_words or token in self.geological_terms
        ]
        
        return filtered_tokens
    
    def _extract_geological_terms(self, tokens: List[str]) -> List[str]:
        """Extract geological terminology from tokens"""
        geological_terms = []
        
        for token in tokens:
            if token in self.geological_terms:
                geological_terms.append(token)
            # Check for compound geological terms
            elif any(term in token for term in ['mineral', 'ore', 'deposit', 'formation']):
                geological_terms.append(token)
        
        return geological_terms
    
    def _extract_mineral_types(self, tokens: List[str]) -> List[str]:
        """Extract mineral types from tokens"""
        mineral_types = []
        
        for token in tokens:
            if token in self.mineral_types:
                mineral_types.append(token)
        
        return mineral_types
    
    def _calculate_text_quality(self, cleaned_text: str, geological_terms: List[str], 
                               mineral_types: List[str]) -> float:
        """Calculate text quality score for geological domain"""
        if not cleaned_text:
            return 0.0
        
        # Base quality from text length
        text_length_score = min(len(cleaned_text.split()) / 50, 1.0)
        
        # Geological term density
        geological_density = len(geological_terms) / max(len(cleaned_text.split()), 1)
        geological_score = min(geological_density * 10, 1.0)
        
        # Mineral type presence
        mineral_score = min(len(mineral_types) / 5, 1.0)
        
        # Technical precision (longer words indicate technical content)
        technical_words = [word for word in cleaned_text.split() if len(word) > 8]
        technical_score = min(len(technical_words) / 10, 1.0)
        
        # Composite quality score
        quality_score = (
            text_length_score * 0.2 +
            geological_score * 0.4 +
            mineral_score * 0.2 +
            technical_score * 0.2
        )
        
        return min(quality_score, 1.0)
    
    def batch_process_texts(self, texts: List[str]) -> List[ProcessedText]:
        """Process multiple geological texts efficiently"""
        processed_texts = []
        
        for text in texts:
            processed = self.process_geological_text(text)
            processed_texts.append(processed)
        
        return processed_texts

class EmbeddingProcessor:
    """
    Geological embedding generation and quality management
    Measurable Success: 10,000+ embeddings with 95%+ quality
    """
    
    def __init__(self, cortex_client: SnowflakeCortexClient):
        self.cortex_client = cortex_client
        self.text_processor = GeologicalTextProcessor()
    
    def generate_geological_embeddings(self, texts: List[str], 
                                     batch_size: Optional[int] = None) -> EmbeddingBatchResult:
        """
        Generate embeddings for geological texts with quality filtering
        Success Metric: 95%+ embedding quality for geological domain
        """
        start_time = time.time()
        
        # Process texts for quality
        processed_texts = self.text_processor.batch_process_texts(texts)
        
        # Filter texts by quality threshold
        quality_threshold = 0.3  # Minimum quality score
        high_quality_texts = [
            pt.cleaned_text for pt in processed_texts 
            if pt.quality_score >= quality_threshold
        ]
        
        if not high_quality_texts:
            logger.warning("No texts met quality threshold for embedding generation")
            return EmbeddingBatchResult(
                total_texts=len(texts),
                successful_embeddings=0,
                failed_embeddings=len(texts),
                average_quality_score=0.0,
                processing_time_seconds=time.time() - start_time,
                embeddings=[]
            )
        
        # Generate embeddings using Cortex
        batch_size = batch_size or config.cortex.max_batch_size
        embeddings = []
        
        for i in range(0, len(high_quality_texts), batch_size):
            batch_texts = high_quality_texts[i:i + batch_size]
            batch_embeddings = self.cortex_client.generate_embeddings_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # Calculate results
        successful_embeddings = sum(1 for emb in embeddings if emb.success)
        failed_embeddings = len(texts) - successful_embeddings
        
        # Calculate average quality score
        quality_scores = [pt.quality_score for pt in processed_texts if pt.quality_score >= quality_threshold]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        processing_time = time.time() - start_time
        
        result = EmbeddingBatchResult(
            total_texts=len(texts),
            successful_embeddings=successful_embeddings,
            failed_embeddings=failed_embeddings,
            average_quality_score=average_quality,
            processing_time_seconds=processing_time,
            embeddings=embeddings
        )
        
        logger.info(f"Generated {successful_embeddings}/{len(texts)} embeddings with {average_quality:.2f} avg quality")
        return result
    
    def process_large_geological_dataset(self, texts: List[str], 
                                       quality_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Process large geological dataset with quality optimization
        Success Metric: 10,000+ high-quality embeddings <10 minutes
        """
        start_time = time.time()
        
        # Process and filter texts
        processed_texts = self.text_processor.batch_process_texts(texts)
        high_quality_texts = [
            pt.cleaned_text for pt in processed_texts 
            if pt.quality_score >= quality_threshold
        ]
        
        # Generate embeddings
        embedding_result = self.cortex_client.batch_process_large_dataset(high_quality_texts)
        
        # Calculate quality metrics
        quality_scores = [pt.quality_score for pt in processed_texts if pt.quality_score >= quality_threshold]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        processing_report = {
            'total_texts_processed': len(texts),
            'high_quality_texts': len(high_quality_texts),
            'quality_threshold': quality_threshold,
            'average_quality_score': average_quality,
            'successful_embeddings': embedding_result['successful_embeddings'],
            'failed_embeddings': embedding_result['failed_embeddings'],
            'success_rate_percentage': embedding_result['success_rate_percentage'],
            'total_processing_time_seconds': time.time() - start_time,
            'performance_target_met': embedding_result['performance_target_met'],
            'quality_distribution': self._calculate_quality_distribution(processed_texts)
        }
        
        return processing_report
    
    def _calculate_quality_distribution(self, processed_texts: List[ProcessedText]) -> Dict[str, int]:
        """Calculate quality score distribution"""
        distribution = {
            'excellent': 0,  # 0.8-1.0
            'good': 0,       # 0.6-0.8
            'fair': 0,       # 0.4-0.6
            'poor': 0,       # 0.2-0.4
            'very_poor': 0   # 0.0-0.2
        }
        
        for pt in processed_texts:
            score = pt.quality_score
            if score >= 0.8:
                distribution['excellent'] += 1
            elif score >= 0.6:
                distribution['good'] += 1
            elif score >= 0.4:
                distribution['fair'] += 1
            elif score >= 0.2:
                distribution['poor'] += 1
            else:
                distribution['very_poor'] += 1
        
        return distribution
    
    def evaluate_embedding_quality(self, test_texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate embedding quality with geological test cases
        Success Metric: 95%+ quality score on geological evaluation dataset
        """
        evaluation_results = {
            'total_test_texts': len(test_texts),
            'quality_scores': [],
            'geological_term_counts': [],
            'mineral_type_counts': [],
            'processing_times': []
        }
        
        for text in test_texts:
            processed = self.text_processor.process_geological_text(text)
            
            evaluation_results['quality_scores'].append(processed.quality_score)
            evaluation_results['geological_term_counts'].append(len(processed.geological_terms))
            evaluation_results['mineral_type_counts'].append(len(processed.mineral_types))
            evaluation_results['processing_times'].append(processed.processing_time_ms)
        
        # Calculate summary statistics
        avg_quality = sum(evaluation_results['quality_scores']) / len(evaluation_results['quality_scores'])
        avg_geological_terms = sum(evaluation_results['geological_term_counts']) / len(evaluation_results['geological_term_counts'])
        avg_mineral_types = sum(evaluation_results['mineral_type_counts']) / len(evaluation_results['mineral_type_counts'])
        avg_processing_time = sum(evaluation_results['processing_times']) / len(evaluation_results['processing_times'])
        
        evaluation_results.update({
            'summary_statistics': {
                'average_quality_score': avg_quality,
                'average_geological_terms': avg_geological_terms,
                'average_mineral_types': avg_mineral_types,
                'average_processing_time_ms': avg_processing_time,
                'quality_target_95_percent_met': avg_quality >= 0.95,
                'processing_speed_target_met': avg_processing_time <= 100  # 100ms per text
            },
            'quality_assessment': {
                'high_quality_texts': sum(1 for score in evaluation_results['quality_scores'] if score >= 0.8),
                'medium_quality_texts': sum(1 for score in evaluation_results['quality_scores'] if 0.6 <= score < 0.8),
                'low_quality_texts': sum(1 for score in evaluation_results['quality_scores'] if score < 0.6)
            }
        })
        
        return evaluation_results
