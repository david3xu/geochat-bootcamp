"""
WebSocket tests
Full Stack AI Engineer Bootcamp - Module 3
"""

import pytest
from django.test import TestCase
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import User
from apps.chat.models import ChatSession
from src.asgi import application
import json

class WebSocketTestCase(TestCase):
    """
    WebSocket test cases
    Learning Outcome: Real-time communication testing
    """
    
    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
        
        # Create test session
        self.session = ChatSession.objects.create(
            user=self.user,
            is_active=True
        )
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.session.id}/"
        )
        communicator.scope['user'] = self.user
        
        connected, _ = await communicator.connect()
        
        self.assertTrue(connected)
        
        # Test connection confirmation message
        response = await communicator.receive_json_from()
        self.assertEqual(response['type'], 'connection_established')
        self.assertEqual(response['session_id'], str(self.session.id))
        
        await communicator.disconnect()
    
    @pytest.mark.asyncio
    async def test_user_message_handling(self):
        """Test user message handling"""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.session.id}/"
        )
        communicator.scope['user'] = self.user
        
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        
        # Send user message
        message_data = {
            'type': 'user_message',
            'message': 'Hello AI',
            'include_spatial_context': True
        }
        
        await communicator.send_json_to(message_data)
        
        # Check user message confirmation
        response = await communicator.receive_json_from()
        self.assertEqual(response['type'], 'user_message_received')
        
        # Check typing indicator
        response = await communicator.receive_json_from()
        self.assertEqual(response['type'], 'ai_typing')
        
        # Check AI response (mock)
        response = await communicator.receive_json_from()
        self.assertEqual(response['type'], 'ai_response')
        
        await communicator.disconnect()
    
    @pytest.mark.asyncio
    async def test_typing_indicator(self):
        """Test typing indicator functionality"""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.session.id}/"
        )
        communicator.scope['user'] = self.user
        
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        
        # Send typing indicator
        typing_data = {
            'type': 'typing_indicator',
            'is_typing': True
        }
        
        await communicator.send_json_to(typing_data)
        
        # Check typing indicator broadcast
        response = await communicator.receive_json_from()
        self.assertEqual(response['type'], 'typing_indicator')
        self.assertTrue(response['is_typing'])
        
        await communicator.disconnect()
    
    @pytest.mark.asyncio
    async def test_invalid_message_handling(self):
        """Test handling of invalid messages"""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.session.id}/"
        )
        communicator.scope['user'] = self.user
        
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        
        # Send invalid JSON
        await communicator.send_to("invalid json")
        
        # Check error response
        response = await communicator.receive_json_from()
        self.assertEqual(response['type'], 'error')
        
        await communicator.disconnect()
    
    @pytest.mark.asyncio
    async def test_authentication_required(self):
        """Test that authentication is required"""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.session.id}/"
        )
        communicator.scope['user'] = None  # No authenticated user
        
        connected, _ = await communicator.connect()
        
        # Should not connect without authentication
        self.assertFalse(connected) 