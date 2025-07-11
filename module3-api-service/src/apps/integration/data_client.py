"""
Data client for Module 1 spatial data integration
Full Stack AI Engineer Bootcamp - Module 3
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional
from django.conf import settings
import time

logger = logging.getLogger('chat2map')

class SpatialDataClient:
    """
    Spatial data client for Module 1 integration
    Learning Outcome: Microservice communication patterns
    
    Measurable Success Criteria:
    - Data retrieval: <500ms response time
    - Spatial queries: Support complex geospatial operations
    - Error handling: Graceful degradation when service unavailable
    """
    
    def __init__(self):
        self.base_url = settings.MODULE1_API_URL
        self.session = None
        self.max_retries = 3
        self.timeout = 10
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def search_relevant_data(self, query: str) -> Dict[str, Any]:
        """
        Search for relevant spatial data based on query
        Learning Outcome: Cross-service data retrieval
        """
        try:
            # Extract spatial keywords from query
            spatial_keywords = self._extract_spatial_keywords(query)
            
            if not spatial_keywords:
                return {'results': [], 'count': 0, 'spatial_context': {}}
            
            # Perform parallel searches
            tasks = []
            
            # Location-based search
            if spatial_keywords.get('locations'):
                tasks.append(self._search_by_location(spatial_keywords['locations']))
            
            # Mineral/geology search
            if spatial_keywords.get('minerals'):
                tasks.append(self._search_by_mineral(spatial_keywords['minerals']))
            
            # General spatial search
            tasks.append(self._search_by_text(query))
            
            # Wait for all searches to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined_results = self._combine_search_results(results)
            
            return {
                'results': combined_results['data'],
                'count': len(combined_results['data']),
                'spatial_context': combined_results['context'],
                'search_types': spatial_keywords
            }
            
        except Exception as e:
            logger.error(f"Error searching spatial data: {e}")
            return {'results': [], 'count': 0, 'spatial_context': {}}
    
    async def _search_by_location(self, locations: List[str]) -> Dict[str, Any]:
        """
        Search by location names
        Learning Outcome: Location-based spatial queries
        """
        try:
            session = await self._get_session()
            
            payload = {
                'locations': locations,
                'limit': 20
            }
            
            async with session.post(
                f"{self.base_url}/spatial/search/location",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Location search failed: {response.status}")
                    return {'results': [], 'search_type': 'location'}
                    
        except Exception as e:
            logger.error(f"Error in location search: {e}")
            return {'results': [], 'search_type': 'location'}
    
    async def _search_by_mineral(self, minerals: List[str]) -> Dict[str, Any]:
        """
        Search by mineral types
        Learning Outcome: Domain-specific data queries
        """
        try:
            session = await self._get_session()
            
            payload = {
                'minerals': minerals,
                'limit': 20
            }
            
            async with session.post(
                f"{self.base_url}/spatial/search/mineral",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Mineral search failed: {response.status}")
                    return {'results': [], 'search_type': 'mineral'}
                    
        except Exception as e:
            logger.error(f"Error in mineral search: {e}")
            return {'results': [], 'search_type': 'mineral'}
    
    async def _search_by_text(self, query: str) -> Dict[str, Any]:
        """
        General text-based search
        Learning Outcome: Full-text search integration
        """
        try:
            session = await self._get_session()
            
            payload = {
                'query': query,
                'limit': 20
            }
            
            async with session.post(
                f"{self.base_url}/spatial/search/text",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Text search failed: {response.status}")
                    return {'results': [], 'search_type': 'text'}
                    
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return {'results': [], 'search_type': 'text'}
    
    def _extract_spatial_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Extract spatial keywords from query
        Learning Outcome: Natural language processing for spatial queries
        """
        query_lower = query.lower()
        
        # Common Australian locations
        locations = []
        location_keywords = [
            'perth', 'melbourne', 'sydney', 'brisbane', 'adelaide', 'darwin',
            'western australia', 'wa', 'victoria', 'vic', 'nsw', 'queensland',
            'qld', 'south australia', 'sa', 'northern territory', 'nt',
            'pilbara', 'kimberley', 'goldfields', 'great western woodlands'
        ]
        
        for keyword in location_keywords:
            if keyword in query_lower:
                locations.append(keyword)
        
        # Common minerals
        minerals = []
        mineral_keywords = [
            'gold', 'iron ore', 'copper', 'nickel', 'zinc', 'lead', 'silver',
            'platinum', 'uranium', 'bauxite', 'coal', 'oil', 'gas', 'lithium',
            'rare earth', 'diamond', 'mineral', 'ore', 'deposit', 'mine'
        ]
        
        for keyword in mineral_keywords:
            if keyword in query_lower:
                minerals.append(keyword)
        
        return {
            'locations': locations,
            'minerals': minerals
        }
    
    def _combine_search_results(self, results: List[Any]) -> Dict[str, Any]:
        """
        Combine results from multiple search types
        Learning Outcome: Data aggregation and deduplication
        """
        combined_data = []
        context = {
            'location_results': 0,
            'mineral_results': 0,
            'text_results': 0,
            'total_sources': 0
        }
        
        seen_ids = set()
        
        for result in results:
            if isinstance(result, Exception):
                continue
                
            if not isinstance(result, dict):
                continue
            
            search_type = result.get('search_type', 'unknown')
            results_data = result.get('results', [])
            
            # Count results by type
            context[f'{search_type}_results'] = len(results_data)
            
            # Add unique results
            for item in results_data:
                item_id = item.get('id')
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    combined_data.append({
                        **item,
                        'search_type': search_type
                    })
        
        # Sort by relevance score if available
        combined_data.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        context['total_sources'] = len(combined_data)
        
        return {
            'data': combined_data[:20],  # Limit to top 20 results
            'context': context
        }
    
    async def get_spatial_details(self, record_id: str) -> Dict[str, Any]:
        """
        Get detailed spatial information for a specific record
        Learning Outcome: Detailed data retrieval patterns
        """
        try:
            session = await self._get_session()
            
            async with session.get(
                f"{self.base_url}/spatial/records/{record_id}",
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Spatial details retrieval failed: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting spatial details: {e}")
            return {}
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
