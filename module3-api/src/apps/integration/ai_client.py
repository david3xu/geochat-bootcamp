"""
AI client for Snowflake Cortex integration
Full Stack AI Engineer Bootcamp - Module 3
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional
from django.conf import settings
import time

logger = logging.getLogger('chat2map')

class SnowflakeCortexClient:
    """
    Snowflake Cortex AI client for natural language processing
    Learning Outcome: Enterprise AI service integration
    
    Measurable Success Criteria:
    - AI response time: <2 seconds average
    - Service availability: 99.9% uptime
    - Error handling: Graceful degradation
    """
    
    def __init__(self):
        self.base_url = settings.MODULE2_AI_URL
        self.cortex_url = settings.SNOWFLAKE_CORTEX_URL
        self.session = None
        self.max_retries = 3
        self.timeout = 30
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process user message with Snowflake Cortex
        Learning Outcome: Asynchronous AI service integration
        """
        start_time = time.time()
        
        try:
            # Generate embeddings for the message
            embedding = await self.generate_embedding(message)
            
            # Perform semantic search
            search_results = await self.semantic_search(embedding)
            
            # Generate AI response
            ai_response = await self.generate_response(message, search_results)
            
            processing_time = time.time() - start_time
            
            return {
                'content': ai_response['content'],
                'confidence': ai_response.get('confidence', 0.0),
                'processing_time': processing_time,
                'search_results': search_results,
                'model_used': 'snowflake-cortex',
                'metadata': {
                    'embedding_generated': True,
                    'search_performed': True,
                    'response_generated': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message with AI: {e}")
            return await self._fallback_response(message)
    
    async def generate_embedding(self, text: str) -> Optional[list]:
        """
        Generate text embeddings using Snowflake Cortex
        Learning Outcome: Vector representation generation
        """
        try:
            session = await self._get_session()
            
            payload = {
                'text': text,
                'model': 'EMBED_TEXT_768'
            }
            
            async with session.post(
                f"{self.base_url}/embeddings/generate",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('embedding')
                else:
                    logger.error(f"Embedding generation failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def semantic_search(self, embedding: list) -> Dict[str, Any]:
        """
        Perform semantic search using embeddings
        Learning Outcome: Vector similarity search
        """
        try:
            if not embedding:
                return {'results': [], 'count': 0}
            
            session = await self._get_session()
            
            payload = {
                'embedding': embedding,
                'limit': 10,
                'similarity_threshold': 0.7
            }
            
            async with session.post(
                f"{self.base_url}/search/semantic",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Semantic search failed: {response.status}")
                    return {'results': [], 'count': 0}
                    
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {'results': [], 'count': 0}
    
    async def generate_response(self, message: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI response using Snowflake Cortex
        Learning Outcome: RAG (Retrieval-Augmented Generation) implementation
        """
        try:
            session = await self._get_session()
            
            # Prepare context from search results
            context = self._prepare_context(search_results)
            
            payload = {
                'message': message,
                'context': context,
                'model': 'CORTEX_COMPLETE',
                'max_tokens': 500,
                'temperature': 0.7
            }
            
            async with session.post(
                f"{self.base_url}/chat/generate",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'content': result.get('response', ''),
                        'confidence': result.get('confidence', 0.0),
                        'context_used': len(context) > 0
                    }
                else:
                    logger.error(f"Response generation failed: {response.status}")
                    return await self._fallback_response(message)
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return await self._fallback_response(message)
    
    def _prepare_context(self, search_results: Dict[str, Any]) -> str:
        """
        Prepare context from search results for RAG
        Learning Outcome: Context preparation for AI responses
        """
        if not search_results.get('results'):
            return ""
        
        context_parts = []
        for result in search_results['results'][:5]:  # Top 5 results
            context_parts.append(f"- {result.get('description', '')}")
        
        return "\n".join(context_parts)
    
    async def _fallback_response(self, message: str) -> Dict[str, Any]:
        """
        Fallback response when AI services are unavailable
        Learning Outcome: Graceful degradation and error handling
        """
        return {
            'content': (
                f"I apologize, but I'm experiencing technical difficulties. "
                f"I received your message about '{message[:50]}...' but cannot "
                f"provide a detailed response at the moment. Please try again shortly."
            ),
            'confidence': 0.0,
            'processing_time': 0.1,
            'search_results': {'results': [], 'count': 0},
            'model_used': 'fallback',
            'metadata': {
                'fallback_used': True,
                'ai_service_available': False
            }
        }
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
