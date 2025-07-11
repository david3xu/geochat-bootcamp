"""
Response aggregator for combining AI and spatial data
Full Stack AI Engineer Bootcamp - Module 3
"""

import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger('chat2map')

class ResponseAggregator:
    """
    Response aggregator for combining AI and spatial responses
    Learning Outcome: Multi-service response coordination
    
    Measurable Success Criteria:
    - Response combination: <50ms processing time
    - Context integration: Meaningful spatial context in AI responses
    - Error handling: Graceful handling of partial service failures
    """
    
    def __init__(self):
        self.context_weight = 0.3
        self.ai_weight = 0.7
    
    async def combine_responses(
        self, 
        ai_response: Dict[str, Any], 
        spatial_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Combine AI and spatial responses into unified response
        Learning Outcome: Multi-modal response integration
        """
        try:
            # Base response from AI
            combined_response = {
                'content': ai_response.get('content', ''),
                'ai_confidence': ai_response.get('confidence', 0.0),
                'processing_time': ai_response.get('processing_time', 0.0),
                'model_used': ai_response.get('model_used', 'unknown'),
                'spatial_context': {},
                'spatial_results_count': 0
            }
            
            # Integrate spatial context if available
            if spatial_response and spatial_response.get('results'):
                spatial_context = self._create_spatial_context(spatial_response)
                enhanced_content = self._enhance_ai_response(
                    ai_response.get('content', ''),
                    spatial_context
                )
                
                combined_response.update({
                    'content': enhanced_content,
                    'spatial_context': spatial_context,
                    'spatial_results_count': spatial_response.get('count', 0)
                })
            
            # Calculate overall confidence
            combined_response['overall_confidence'] = self._calculate_overall_confidence(
                ai_response.get('confidence', 0.0),
                spatial_response.get('count', 0) if spatial_response else 0
            )
            
            return combined_response
            
        except Exception as e:
            logger.error(f"Error combining responses: {e}")
            return {
                'content': ai_response.get('content', 'Error processing response'),
                'ai_confidence': 0.0,
                'overall_confidence': 0.0,
                'spatial_context': {},
                'spatial_results_count': 0,
                'error': str(e)
            }
    
    def _create_spatial_context(self, spatial_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create spatial context from search results
        Learning Outcome: Spatial data summarization
        """
        results = spatial_response.get('results', [])
        context = spatial_response.get('spatial_context', {})
        
        if not results:
            return {}
        
        # Extract relevant locations
        locations = set()
        minerals = set()
        regions = set()
        
        for result in results[:5]:  # Top 5 results
            # Extract location information
            location = result.get('location', {})
            if location.get('name'):
                locations.add(location['name'])
            if location.get('region'):
                regions.add(location['region'])
            
            # Extract mineral information
            if result.get('mineral_type'):
                minerals.add(result['mineral_type'])
        
        return {
            'locations': list(locations),
            'minerals': list(minerals),
            'regions': list(regions),
            'total_records': len(results),
            'search_context': context,
            'top_result': results[0] if results else None
        }
    
    def _enhance_ai_response(self, ai_content: str, spatial_context: Dict[str, Any]) -> str:
        """
        Enhance AI response with spatial context
        Learning Outcome: Context-aware response generation
        """
        try:
            # Check if AI response already includes spatial information
            if any(location.lower() in ai_content.lower() 
                   for location in spatial_context.get('locations', [])):
                return ai_content
            
            # Add spatial context to response
            enhancement = self._generate_spatial_enhancement(spatial_context)
            
            if enhancement:
                enhanced_content = f"{ai_content}\n\n{enhancement}"
                return enhanced_content
            
            return ai_content
            
        except Exception as e:
            logger.error(f"Error enhancing AI response: {e}")
            return ai_content
    
    def _generate_spatial_enhancement(self, spatial_context: Dict[str, Any]) -> str:
        """
        Generate spatial enhancement text
        Learning Outcome: Dynamic content generation
        """
        try:
            enhancements = []
            
            # Location information
            locations = spatial_context.get('locations', [])
            if locations:
                location_text = ", ".join(locations[:3])  # Top 3 locations
                enhancements.append(f"📍 **Relevant locations**: {location_text}")
            
            # Mineral information
            minerals = spatial_context.get('minerals', [])
            if minerals:
                mineral_text = ", ".join(minerals[:3])  # Top 3 minerals
                enhancements.append(f"⛏️ **Related minerals**: {mineral_text}")
            
            # Data context
            total_records = spatial_context.get('total_records', 0)
            if total_records > 0:
                enhancements.append(f"📊 **Found {total_records} relevant geological records**")
            
            # Top result details
            top_result = spatial_context.get('top_result')
            if top_result:
                description = top_result.get('description', '')
                if description:
                    enhancements.append(f"🎯 **Most relevant**: {description[:100]}...")
            
            return "\n".join(enhancements) if enhancements else ""
            
        except Exception as e:
            logger.error(f"Error generating spatial enhancement: {e}")
            return ""
    
    def _calculate_overall_confidence(self, ai_confidence: float, spatial_count: int) -> float:
        """
        Calculate overall confidence score
        Learning Outcome: Multi-factor confidence scoring
        """
        try:
            # Base confidence from AI
            base_confidence = ai_confidence * self.ai_weight
            
            # Spatial data confidence (based on result count)
            spatial_confidence = min(spatial_count / 10.0, 1.0) * self.context_weight
            
            # Combined confidence
            overall_confidence = base_confidence + spatial_confidence
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
