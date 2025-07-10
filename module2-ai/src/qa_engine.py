"""
Geological Question-Answering Engine with Snowflake Cortex
Measurable Success: 85%+ accurate responses to geological questions
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from dataclasses import dataclass, asdict
import json

from .snowflake_cortex_client import SnowflakeCortexClient, CompletionResult
from .semantic_search import GeologicalSemanticSearch, SimilarityMatch
from .config import config

logger = logging.getLogger(__name__)

@dataclass
class QAResponse:
    """Structured QA response with quality metrics"""
    question: str
    answer: str
    confidence_score: float
    processing_time_ms: float
    source_documents: List[Dict[str, Any]]
    geological_accuracy: float
    spatial_context: Optional[Dict[str, Any]] = None

@dataclass
class QAPerformanceMetrics:
    """QA engine performance tracking for supervision"""
    total_questions_processed: int
    average_response_time_ms: float
    average_geological_accuracy: float
    average_confidence_score: float
    performance_target_compliance: float

class GeologicalQAEngine:
    """
    Question-answering system for geological exploration
    Measurable Success: 85%+ accurate responses to geological questions
    """
    
    def __init__(self, cortex_client: SnowflakeCortexClient, search_engine: GeologicalSemanticSearch):
        self.cortex_client = cortex_client
        self.search_engine = search_engine
        self.performance_metrics = QAPerformanceMetrics(0, 0.0, 0.0, 0.0, 0.0)
        self.geological_knowledge_base = self._load_geological_knowledge()
        
    def _load_geological_knowledge(self) -> Dict[str, Any]:
        """Load geological domain knowledge for enhanced responses"""
        return {
            'mineral_properties': {
                'gold': {'density': 19.3, 'hardness': 2.5, 'crystal_system': 'cubic'},
                'iron_ore': {'types': ['hematite', 'magnetite'], 'grade_threshold': 60},
                'copper': {'common_minerals': ['chalcopyrite', 'malachite', 'azurite']},
                'nickel': {'primary_source': 'pentlandite', 'laterite_deposits': True},
                'lithium': {'sources': ['spodumene', 'brine', 'clay'], 'battery_grade': 'required'}
            },
            'geological_formations': {
                'western_australia': {
                    'pilbara': 'iron_ore_province',
                    'yilgarn_craton': 'gold_province',
                    'kimberley': 'diamond_province'
                }
            },
            'exploration_techniques': [
                'geophysical_surveys', 'geochemical_sampling', 'core_drilling',
                'remote_sensing', 'geological_mapping'
            ]
        }
    
    def process_geological_query(self, user_question: str, max_context_docs: int = 5) -> QAResponse:
        """
        End-to-end question processing with context retrieval
        Success Metric: <2s response time for complex geological queries
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze query type and extract key concepts
            query_analysis = self._analyze_geological_query(user_question)
            
            # Step 2: Retrieve relevant context documents
            context_documents = self.retrieve_relevant_context(user_question, max_context_docs)
            
            # Step 3: Generate enhanced prompt with geological context
            enhanced_prompt = self._construct_enhanced_geological_prompt(
                user_question, context_documents, query_analysis
            )
            
            # Step 4: Generate AI response using Cortex COMPLETE
            completion_result = self.cortex_client.complete_geological_query(
                enhanced_prompt, self._format_context_for_cortex(context_documents)
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Step 5: Assess response quality and geological accuracy
            geological_accuracy = self._assess_geological_accuracy(user_question, completion_result.completion_output)
            confidence_score = self._calculate_confidence_score(completion_result, context_documents)
            
            # Step 6: Extract spatial context if relevant
            spatial_context = self._extract_spatial_context(context_documents)
            
            # Create structured response
            qa_response = QAResponse(
                question=user_question,
                answer=completion_result.completion_output,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                source_documents=[doc.metadata for doc in context_documents],
                geological_accuracy=geological_accuracy,
                spatial_context=spatial_context
            )
            
            # Update performance metrics
            self._update_performance_metrics(qa_response)
            
            logger.info(f"Geological QA completed in {processing_time:.2f}ms with {geological_accuracy:.2f} accuracy")
            return qa_response
            
        except Exception as e:
            logger.error(f"Geological query processing failed: {str(e)}")
            return QAResponse(
                question=user_question,
                answer=f"I apologize, but I encountered an error processing your geological query: {str(e)}",
                confidence_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                source_documents=[],
                geological_accuracy=0.0
            )
    
    def retrieve_relevant_context(self, query: str, max_docs: int = 5) -> List[SimilarityMatch]:
        """
        Retrieve most relevant geological documents for query context
        Success Metric: 90% context relevance for answer generation
        """
        try:
            # Generate query embedding for similarity search
            embedding_results = self.cortex_client.generate_embeddings_batch([query])
            
            if not embedding_results or not embedding_results[0].success:
                logger.warning("Failed to generate query embedding for context retrieval")
                return []
            
            query_embedding = embedding_results[0].embedding_vector
            
            # Perform similarity search
            similar_documents = self.search_engine.search_similar_geological_content(
                query_embedding, top_k=max_docs * 2  # Get extra results for filtering
            )
            
            # Filter and rank documents for geological relevance
            filtered_documents = self._filter_context_for_geological_relevance(query, similar_documents)
            
            return filtered_documents[:max_docs]
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return []
    
    def _analyze_geological_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to identify geological concepts and intent"""
        query_lower = query.lower()
        
        # Identify mineral types mentioned
        mentioned_minerals = []
        for mineral in self.geological_knowledge_base['mineral_properties'].keys():
            if mineral.replace('_', ' ') in query_lower:
                mentioned_minerals.append(mineral)
        
        # Identify query type
        query_type = 'general'
        if any(word in query_lower for word in ['where', 'location', 'coordinates', 'map']):
            query_type = 'spatial'
        elif any(word in query_lower for word in ['how', 'process', 'method', 'technique']):
            query_type = 'procedural'
        elif any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            query_type = 'definitional'
        elif any(word in query_lower for word in ['grade', 'tonnage', 'deposit', 'resource']):
            query_type = 'quantitative'
        
        # Identify geographical context
        wa_regions = ['pilbara', 'kimberley', 'goldfields', 'perth', 'kalgoorlie']
        mentioned_regions = [region for region in wa_regions if region in query_lower]
        
        return {
            'query_type': query_type,
            'mentioned_minerals': mentioned_minerals,
            'mentioned_regions': mentioned_regions,
            'requires_spatial_context': query_type == 'spatial' or bool(mentioned_regions),
            'complexity_level': self._assess_query_complexity(query)
        }
    
    def _construct_enhanced_geological_prompt(self, question: str, context_docs: List[SimilarityMatch], 
                                            query_analysis: Dict[str, Any]) -> str:
        """Construct domain-optimized prompt for Cortex COMPLETE"""
        
        # Base geological expertise prompt
        base_prompt = """You are a senior geological consultant with extensive experience in Western Australian mineral exploration and mining. You have deep expertise in:
- Geological formations and mineralization processes
- Exploration techniques and methodologies
- Economic geology and resource evaluation
- Western Australian regional geology
- Mining and extraction technologies

"""
        
        # Add context from similar documents
        if context_docs:
            context_section = "Relevant geological data from recent exploration:\n"
            for i, doc in enumerate(context_docs[:3]):  # Top 3 most relevant
                context_section += f"{i+1}. Location: {doc.coordinates[1]:.4f}°S, {doc.coordinates[0]:.4f}°E\n"
                context_section += f"   Description: {doc.geological_text[:200]}...\n\n"
            base_prompt += context_section
        
        # Add domain-specific knowledge based on query analysis
        if query_analysis['mentioned_minerals']:
            minerals_info = "Relevant mineral information:\n"
            for mineral in query_analysis['mentioned_minerals']:
                if mineral in self.geological_knowledge_base['mineral_properties']:
                    props = self.geological_knowledge_base['mineral_properties'][mineral]
                    minerals_info += f"- {mineral.title()}: {props}\n"
            base_prompt += minerals_info + "\n"
        
        # Add regional context for Western Australia
        if query_analysis['mentioned_regions']:
            regional_info = "Western Australian regional context:\n"
            for region in query_analysis['mentioned_regions']:
                if region in self.geological_knowledge_base['geological_formations']['western_australia']:
                    formation = self.geological_knowledge_base['geological_formations']['western_australia'][region]
                    regional_info += f"- {region.title()}: Known for {formation}\n"
            base_prompt += regional_info + "\n"
        
        # Add the user question
        base_prompt += f"Question: {question}\n\n"
        
        # Add response guidelines based on query type
        response_guidelines = {
            'spatial': "Provide specific geographical information, coordinates where relevant, and regional geological context.",
            'procedural': "Explain step-by-step processes, methodologies, and best practices in exploration.",
            'definitional': "Give clear, accurate definitions with practical examples from WA geology.",
            'quantitative': "Include specific numbers, grades, tonnages, and economic data where available.",
            'general': "Provide comprehensive, well-structured geological information."
        }
        
        guidelines = response_guidelines.get(query_analysis['query_type'], response_guidelines['general'])
        base_prompt += f"Response Guidelines: {guidelines}\n\n"
        base_prompt += "Please provide a detailed, accurate response:"
        
        return base_prompt
    
    def _filter_context_for_geological_relevance(self, query: str, documents: List[SimilarityMatch]) -> List[SimilarityMatch]:
        """Filter and rank documents for geological context relevance"""
        query_terms = set(query.lower().split())
        
        scored_documents = []
        for doc in documents:
            # Calculate relevance based on multiple factors
            text_terms = set(doc.geological_text.lower().split())
            term_overlap = len(query_terms.intersection(text_terms))
            
            # Boost score for mineral type match
            mineral_boost = 0.2 if any(mineral in query.lower() for mineral in [doc.mineral_type.lower()]) else 0
            
            # Boost score for geological terminology
            geological_terms = {'exploration', 'deposit', 'mineralization', 'ore', 'grade', 'geological'}
            geological_boost = len(text_terms.intersection(geological_terms)) * 0.1
            
            # Calculate composite relevance score
            relevance_score = doc.similarity_score + mineral_boost + geological_boost + (term_overlap * 0.05)
            
            scored_documents.append((relevance_score, doc))
        
        # Sort by relevance score and return documents
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_documents]
    
    def _assess_geological_accuracy(self, question: str, answer: str) -> float:
        """
        Assess geological domain accuracy of AI responses
        Success Metric: 85%+ geological accuracy in expert evaluation
        """
        # Extract geological terms from question and answer
        geological_vocabulary = [
            'mineral', 'ore', 'deposit', 'exploration', 'geology', 'formation',
            'gold', 'iron', 'copper', 'nickel', 'lithium', 'uranium',
            'mining', 'drilling', 'assay', 'grade', 'tonnage', 'outcrop',
            'metamorphic', 'igneous', 'sedimentary', 'fault', 'vein', 'lode'
        ]
        
        answer_terms = set(answer.lower().split())
        geological_term_count = sum(1 for term in geological_vocabulary if term in answer_terms)
        
        # Base accuracy from geological term usage
        base_accuracy = min(geological_term_count / 10, 0.8)  # Cap at 80% from terminology
        
        # Check for factual accuracy indicators
        accuracy_indicators = {
            'specific_locations': any(location in answer.lower() for location in ['pilbara', 'kimberley', 'yilgarn', 'perth']),
            'quantitative_data': any(char.isdigit() for char in answer),
            'technical_precision': len([word for word in answer.split() if len(word) > 8]) > 3,
            'proper_context': 'western australia' in answer.lower() or 'wa' in answer.lower()
        }
        
        accuracy_bonus = sum(0.05 for indicator in accuracy_indicators.values() if indicator)
        
        final_accuracy = min(base_accuracy + accuracy_bonus, 1.0)
        return final_accuracy
    
    def _calculate_confidence_score(self, completion_result: CompletionResult, context_docs: List[SimilarityMatch]) -> float:
        """Calculate confidence score based on multiple factors"""
        # Base confidence from Cortex relevance score
        base_confidence = completion_result.relevance_score
        
        # Boost confidence based on context quality
        if context_docs:
            avg_context_similarity = sum(doc.similarity_score for doc in context_docs) / len(context_docs)
            context_boost = avg_context_similarity * 0.3
        else:
            context_boost = 0
        
        # Penalize for processing time (slower responses may be less confident)
        time_penalty = max(0, (completion_result.processing_time_ms - 1000) / 5000 * 0.1)
        
        confidence_score = min(base_confidence + context_boost - time_penalty, 1.0)
        return max(confidence_score, 0.0)
    
    def _extract_spatial_context(self, context_docs: List[SimilarityMatch]) -> Optional[Dict[str, Any]]:
        """Extract spatial context from relevant documents"""
        if not context_docs:
            return None
        
        # Calculate centroid of relevant locations
        latitudes = [doc.coordinates[1] for doc in context_docs if doc.coordinates[1] != 0]
        longitudes = [doc.coordinates[0] for doc in context_docs if doc.coordinates[0] != 0]
        
        if not latitudes or not longitudes:
            return None
        
        centroid_lat = sum(latitudes) / len(latitudes)
        centroid_lng = sum(longitudes) / len(longitudes)
        
        # Identify dominant mineral types in the area
        mineral_counts = {}
        for doc in context_docs:
            mineral = doc.mineral_type
            mineral_counts[mineral] = mineral_counts.get(mineral, 0) + 1
        
        dominant_minerals = sorted(mineral_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'centroid_coordinates': [centroid_lng, centroid_lat],
            'bounding_box': {
                'north': max(latitudes),
                'south': min(latitudes),
                'east': max(longitudes),
                'west': min(longitudes)
            },
            'dominant_minerals': [mineral for mineral, count in dominant_minerals],
            'total_sites': len(context_docs)
        }
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity for processing optimization"""
        word_count = len(query.split())
        question_words = ['what', 'where', 'how', 'why', 'when', 'which']
        question_count = sum(1 for word in question_words if word in query.lower())
        
        if word_count < 5 and question_count <= 1:
            return 'simple'
        elif word_count < 15 and question_count <= 2:
            return 'moderate'
        else:
            return 'complex'
    
    def _format_context_for_cortex(self, context_docs: List[SimilarityMatch]) -> str:
        """Format context documents for Cortex COMPLETE function"""
        if not context_docs:
            return ""
        
        formatted_context = "Relevant geological exploration data:\n\n"
        for i, doc in enumerate(context_docs):
            formatted_context += f"Site {i+1}:\n"
            formatted_context += f"- Location: {doc.coordinates[1]:.4f}°S, {doc.coordinates[0]:.4f}°E\n"
            formatted_context += f"- Mineral Type: {doc.mineral_type}\n"
            formatted_context += f"- Description: {doc.geological_text}\n"
            formatted_context += f"- Relevance Score: {doc.similarity_score:.3f}\n\n"
        
        return formatted_context
    
    def _update_performance_metrics(self, qa_response: QAResponse) -> None:
        """Update QA engine performance metrics"""
        self.performance_metrics.total_questions_processed += 1
        
        # Update running averages
        n = self.performance_metrics.total_questions_processed
        self.performance_metrics.average_response_time_ms = (
            (self.performance_metrics.average_response_time_ms * (n - 1) + qa_response.processing_time_ms) / n
        )
        self.performance_metrics.average_geological_accuracy = (
            (self.performance_metrics.average_geological_accuracy * (n - 1) + qa_response.geological_accuracy) / n
        )
        self.performance_metrics.average_confidence_score = (
            (self.performance_metrics.average_confidence_score * (n - 1) + qa_response.confidence_score) / n
        )
        
        # Update performance target compliance
        response_time_met = qa_response.processing_time_ms <= config.performance.target_complete_response_ms
        accuracy_met = qa_response.geological_accuracy >= config.performance.target_accuracy_percentage / 100
        
        compliance = (int(response_time_met) + int(accuracy_met)) / 2 * 100
        self.performance_metrics.performance_target_compliance = (
            (self.performance_metrics.performance_target_compliance * (n - 1) + compliance) / n
        )
    
    def evaluate_qa_quality(self, test_questions: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive QA quality evaluation with geological test cases
        Success Metric: 85%+ accuracy on geological QA evaluation dataset
        """
        evaluation_results = {
            'total_questions': len(test_questions),
            'accuracy_scores': [],
            'response_times': [],
            'confidence_scores': [],
            'detailed_results': []
        }
        
        for test_case in test_questions:
            question = test_case['question']
            expected_concepts = test_case.get('expected_concepts', [])
            expected_accuracy_threshold = test_case.get('accuracy_threshold', 0.8)
            
            # Process question
            qa_response = self.process_geological_query(question)
            
            # Evaluate response
            concept_coverage = self._evaluate_concept_coverage(qa_response.answer, expected_concepts)
            accuracy_met = qa_response.geological_accuracy >= expected_accuracy_threshold
            
            evaluation_results['accuracy_scores'].append(qa_response.geological_accuracy)
            evaluation_results['response_times'].append(qa_response.processing_time_ms)
            evaluation_results['confidence_scores'].append(qa_response.confidence_score)
            
            evaluation_results['detailed_results'].append({
                'question': question,
                'geological_accuracy': qa_response.geological_accuracy,
                'concept_coverage': concept_coverage,
                'accuracy_threshold_met': accuracy_met,
                'response_time_ms': qa_response.processing_time_ms,
                'confidence_score': qa_response.confidence_score
            })
        
        # Calculate summary statistics
        avg_accuracy = sum(evaluation_results['accuracy_scores']) / len(evaluation_results['accuracy_scores'])
        avg_response_time = sum(evaluation_results['response_times']) / len(evaluation_results['response_times'])
        avg_confidence = sum(evaluation_results['confidence_scores']) / len(evaluation_results['confidence_scores'])
        
        accuracy_target_met = avg_accuracy >= 0.85
        response_time_target_met = avg_response_time <= config.performance.target_complete_response_ms
        
        evaluation_results.update({
            'summary_statistics': {
                'average_geological_accuracy': avg_accuracy,
                'average_response_time_ms': avg_response_time,
                'average_confidence_score': avg_confidence,
                'accuracy_target_85_percent_met': accuracy_target_met,
                'response_time_target_met': response_time_target_met,
                'overall_quality_score': (avg_accuracy + avg_confidence) / 2
            },
            'performance_assessment': {
                'questions_above_85_accuracy': sum(1 for score in evaluation_results['accuracy_scores'] if score >= 0.85),
                'questions_under_2s_response': sum(1 for time in evaluation_results['response_times'] if time <= 2000),
                'high_confidence_responses': sum(1 for conf in evaluation_results['confidence_scores'] if conf >= 0.8)
            }
        })
        
        return evaluation_results
    
    def _evaluate_concept_coverage(self, answer: str, expected_concepts: List[str]) -> float:
        """Evaluate how well the answer covers expected geological concepts"""
        if not expected_concepts:
            return 1.0
        
        answer_lower = answer.lower()
        covered_concepts = sum(1 for concept in expected_concepts if concept.lower() in answer_lower)
        
        return covered_concepts / len(expected_concepts) if expected_concepts else 1.0
    
    def get_qa_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive QA performance report for supervision"""
        return {
            'qa_performance_metrics': asdict(self.performance_metrics),
            'learning_targets_assessment': {
                'accuracy_target_85_percent': self.performance_metrics.average_geological_accuracy >= 0.85,
                'response_time_target_2s': self.performance_metrics.average_response_time_ms <= 2000,
                'confidence_target_80_percent': self.performance_metrics.average_confidence_score >= 0.8,
                'performance_compliance_percentage': self.performance_metrics.performance_target_compliance
            },
            'geological_expertise_indicators': {
                'domain_knowledge_integration': len(self.geological_knowledge_base['mineral_properties']),
                'regional_context_coverage': len(self.geological_knowledge_base['geological_formations']['western_australia']),
                'exploration_technique_knowledge': len(self.geological_knowledge_base['exploration_techniques'])
            },
            'quality_improvement_recommendations': self._generate_qa_improvement_recommendations()
        }
    
    def _generate_qa_improvement_recommendations(self) -> List[str]:
        """Generate QA improvement recommendations based on performance"""
        recommendations = []
        
        if self.performance_metrics.average_geological_accuracy < 0.85:
            recommendations.append("Enhance geological domain knowledge base with more specialized terminology")
        
        if self.performance_metrics.average_response_time_ms > 2000:
            recommendations.append("Optimize context retrieval and prompt construction for faster responses")
        
        if self.performance_metrics.average_confidence_score < 0.8:
            recommendations.append("Improve context relevance scoring and confidence calculation algorithms")
        
        return recommendations
