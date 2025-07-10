"""
Integration tests
Full Stack AI Engineer Bootcamp - Module 3
"""

import pytest
from django.test import TestCase
from unittest.mock import patch, AsyncMock
from apps.integration.ai_client import SnowflakeCortexClient
from apps.integration.data_client import SpatialDataClient
from apps.integration.response_aggregator import ResponseAggregator

class IntegrationTestCase(TestCase):
    """
    Integration test cases
    Learning Outcome: Multi-service integration testing
    """
    
    def setUp(self):
        """Set up test clients"""
        self.ai_client = SnowflakeCortexClient()
        self.spatial_client = SpatialDataClient()
        self.response_aggregator = ResponseAggregator()
    
    @pytest.mark.asyncio
    async def test_ai_client_message_processing(self):
        """Test AI client message processing"""
        with patch('apps.integration.ai_client.aiohttp.ClientSession') as mock_session:
            # Mock successful AI response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'response': 'This is a test AI response',
                'confidence': 0.85
            }
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await self.ai_client.process_message("Test message")
            
            self.assertIn('content', result)
            self.assertIn('confidence', result)
            self.assertEqual(result['model_used'], 'snowflake-cortex')
    
    @pytest.mark.asyncio
    async def test_spatial_client_search(self):
        """Test spatial client search functionality"""
        with patch('apps.integration.data_client.aiohttp.ClientSession') as mock_session:
            # Mock successful spatial search
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'results': [
                    {
                        'id': 'geo_001',
                        'name': 'Test Location',
                        'location': {'lat': -31.9505, 'lng': 115.8605},
                        'mineral_type': 'Gold'
                    }
                ],
                'count': 1
            }
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await self.spatial_client.search_relevant_data("gold mining in Western Australia")
            
            self.assertIn('results', result)
            self.assertIn('count', result)
            self.assertEqual(result['count'], 1)
    
    @pytest.mark.asyncio
    async def test_response_aggregator(self):
        """Test response aggregator functionality"""
        # Mock AI response
        ai_response = {
            'content': 'This is an AI response',
            'confidence': 0.8,
            'processing_time': 1.5
        }
        
        # Mock spatial response
        spatial_response = {
            'results': [
                {
                    'id': 'geo_001',
                    'name': 'Gold Mine',
                    'location': {'name': 'Western Australia'},
                    'mineral_type': 'Gold'
                }
            ],
            'count': 1
        }
        
        # Test response aggregation
        result = await self.response_aggregator.combine_responses(ai_response, spatial_response)
        
        self.assertIn('content', result)
        self.assertIn('spatial_context', result)
        self.assertIn('overall_confidence', result)
        self.assertGreater(result['overall_confidence'], 0)
    
    @pytest.mark.asyncio
    async def test_ai_client_fallback(self):
        """Test AI client fallback when service is unavailable"""
        with patch('apps.integration.ai_client.aiohttp.ClientSession') as mock_session:
            # Mock failed AI response
            mock_response = AsyncMock()
            mock_response.status = 500
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await self.ai_client.process_message("Test message")
            
            self.assertIn('content', result)
            self.assertEqual(result['model_used'], 'fallback')
            self.assertEqual(result['confidence'], 0.0)
    
    @pytest.mark.asyncio
    async def test_spatial_keyword_extraction(self):
        """Test spatial keyword extraction"""
        # Test location extraction
        query = "Tell me about gold mining in Perth, Western Australia"
        keywords = self.spatial_client._extract_spatial_keywords(query)
        
        self.assertIn('perth', keywords['locations'])
        self.assertIn('western australia', keywords['locations'])
        self.assertIn('gold', keywords['minerals'])
    
    def test_confidence_calculation(self):
        """Test confidence calculation in response aggregator"""
        # Test with high AI confidence and spatial results
        ai_confidence = 0.9
        spatial_count = 5
        
        overall_confidence = self.response_aggregator._calculate_overall_confidence(
            ai_confidence, spatial_count
        )
        
        self.assertGreater(overall_confidence, 0.8)
        self.assertLessEqual(overall_confidence, 1.0)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in integration components"""
        # Test AI client error handling
        with patch('apps.integration.ai_client.aiohttp.ClientSession') as mock_session:
            mock_session.side_effect = Exception("Connection failed")
            
            result = await self.ai_client.process_message("Test message")
            
            self.assertIn('content', result)
            self.assertEqual(result['model_used'], 'fallback')
        
        # Test spatial client error handling
        with patch('apps.integration.data_client.aiohttp.ClientSession') as mock_session:
            mock_session.side_effect = Exception("Connection failed")
            
            result = await self.spatial_client.search_relevant_data("Test query")
            
            self.assertIn('results', result)
            self.assertEqual(result['count'], 0)
