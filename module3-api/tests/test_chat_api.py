"""
Chat API tests
Full Stack AI Engineer Bootcamp - Module 3
"""

import pytest
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth.models import User
from apps.chat.models import ChatSession, ChatMessage, UserQuery
import uuid

class ChatAPITestCase(TestCase):
    """
    Chat API test cases
    Learning Outcome: API testing and validation
    """
    
    def setUp(self):
        """Set up test data"""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
        self.client.force_authenticate(user=self.user)
        
        # Create test session
        self.session = ChatSession.objects.create(
            user=self.user,
            is_active=True
        )
    
    def test_create_chat_session(self):
        """Test creating a new chat session"""
        url = reverse('chatsession-list')
        response = self.client.post(url, {})
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(ChatSession.objects.count(), 2)  # Including setUp session
    
    def test_get_chat_sessions(self):
        """Test retrieving chat sessions"""
        url = reverse('chatsession-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
    
    def test_end_chat_session(self):
        """Test ending a chat session"""
        url = reverse('chatsession-end-session', args=[self.session.id])
        response = self.client.post(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.session.refresh_from_db()
        self.assertFalse(self.session.is_active)
    
    def test_get_session_messages(self):
        """Test retrieving session messages"""
        # Create test messages
        ChatMessage.objects.create(
            session=self.session,
            message_type='user',
            content='Hello'
        )
        ChatMessage.objects.create(
            session=self.session,
            message_type='ai',
            content='Hi there!'
        )
        
        url = reverse('chatmessage-session-messages')
        response = self.client.get(url, {'session_id': str(self.session.id)})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
    
    def test_search_messages(self):
        """Test searching messages"""
        # Create test messages
        ChatMessage.objects.create(
            session=self.session,
            message_type='user',
            content='Hello world'
        )
        
        url = reverse('chatmessage-search-messages')
        response = self.client.get(url, {'q': 'world'})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 1)
    
    def test_analytics_endpoints(self):
        """Test analytics endpoints"""
        # Create test query
        UserQuery.objects.create(
            session=self.session,
            query_text='Test query',
            response_time=150.0,
            spatial_results_count=5,
            ai_confidence_score=0.85
        )
        
        # Test response time analytics
        url = reverse('analytics-response-time-analytics')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_queries', response.data)
        
        # Test session analytics
        url = reverse('analytics-session-analytics')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_sessions', response.data)
