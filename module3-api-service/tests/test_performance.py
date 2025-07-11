"""
Performance tests
Full Stack AI Engineer Bootcamp - Module 3
"""

import pytest
import time
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth.models import User
from apps.chat.models import ChatSession, ChatMessage
from apps.monitoring.health_checks import HealthCheckManager
from apps.monitoring.metrics import MetricsCollector

class PerformanceTestCase(TestCase):
    """
    Performance test cases
    Learning Outcome: Performance testing and benchmarking
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
    
    def test_api_response_time(self):
        """Test API response time performance"""
        # Test chat session creation performance
        start_time = time.time()
        url = reverse('chatsession-list')
        response = self.client.post(url, {})
        end_time = time.time()
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Performance benchmark: <100ms for session creation
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        self.assertLess(response_time, 100, f"Response time {response_time}ms exceeds 100ms limit")
        
        # Test session retrieval performance
        start_time = time.time()
        response = self.client.get(url)
        end_time = time.time()
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Performance benchmark: <200ms for session retrieval
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 200, f"Response time {response_time}ms exceeds 200ms limit")
    
    def test_message_search_performance(self):
        """Test message search performance"""
        # Create test messages
        for i in range(50):
            ChatMessage.objects.create(
                session=self.session,
                message_type='user' if i % 2 == 0 else 'ai',
                content=f'Test message {i} with searchable content'
            )
        
        # Test search performance
        start_time = time.time()
        url = reverse('chatmessage-search-messages')
        response = self.client.get(url, {'q': 'searchable'})
        end_time = time.time()
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Performance benchmark: <300ms for search
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 300, f"Search time {response_time}ms exceeds 300ms limit")
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance"""
        health_manager = HealthCheckManager()
        
        start_time = time.time()
        health_data = await health_manager.run_all_checks()
        end_time = time.time()
        
        # Performance benchmark: <1 second for all health checks
        response_time = end_time - start_time
        self.assertLess(response_time, 1.0, f"Health check time {response_time}s exceeds 1s limit")
        
        # Verify health check structure
        self.assertIn('overall_status', health_data)
        self.assertIn('checks', health_data)
        self.assertIn('total_response_time', health_data)
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance"""
        metrics_collector = MetricsCollector()
        
        start_time = time.time()
        metrics_data = metrics_collector.get_metrics('1h')
        end_time = time.time()
        
        # Performance benchmark: <500ms for metrics collection
        response_time = (end_time - start_time) * 1000
        self.assertLess(response_time, 500, f"Metrics collection time {response_time}ms exceeds 500ms limit")
        
        # Verify metrics structure
        self.assertIn('period', metrics_data)
        self.assertIn('api_metrics', metrics_data)
        self.assertIn('websocket_metrics', metrics_data)
    
    def test_concurrent_session_creation(self):
        """Test concurrent session creation performance"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def create_session():
            """Create a session and record response time"""
            start_time = time.time()
            url = reverse('chatsession-list')
            response = self.client.post(url, {})
            end_time = time.time()
            
            results.put({
                'status_code': response.status_code,
                'response_time': (end_time - start_time) * 1000
            })
        
        # Create 10 concurrent sessions
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_session)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        response_times = []
        for _ in range(10):
            result = results.get()
            self.assertEqual(result['status_code'], status.HTTP_201_CREATED)
            response_times.append(result['response_time'])
        
        # Performance benchmark: Average response time <150ms under concurrent load
        avg_response_time = sum(response_times) / len(response_times)
        self.assertLess(avg_response_time, 150, f"Average concurrent response time {avg_response_time}ms exceeds 150ms limit")
    
    def test_database_query_performance(self):
        """Test database query performance"""
        # Create test data
        for i in range(100):
            ChatMessage.objects.create(
                session=self.session,
                message_type='user' if i % 2 == 0 else 'ai',
                content=f'Performance test message {i}',
                processing_time=100 + i
            )
        
        # Test query performance
        start_time = time.time()
        messages = ChatMessage.objects.filter(session=self.session).order_by('-timestamp')[:25]
        end_time = time.time()
        
        # Performance benchmark: <50ms for database query
        query_time = (end_time - start_time) * 1000
        self.assertLess(query_time, 50, f"Database query time {query_time}ms exceeds 50ms limit")
        
        # Verify query results
        self.assertEqual(len(messages), 25)
    
    def test_memory_usage(self):
        """Test memory usage during operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        for i in range(1000):
            ChatMessage.objects.create(
                session=self.session,
                message_type='user',
                content=f'Memory test message {i} with some content to increase memory usage'
            )
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Performance benchmark: Memory increase <50MB for 1000 operations
        self.assertLess(memory_increase, 50, f"Memory increase {memory_increase}MB exceeds 50MB limit")
    
    def test_error_rate_under_load(self):
        """Test error rate under load"""
        error_count = 0
        total_requests = 100
        
        for i in range(total_requests):
            try:
                url = reverse('chatsession-list')
                response = self.client.get(url)
                if response.status_code >= 400:
                    error_count += 1
            except Exception:
                error_count += 1
        
        error_rate = (error_count / total_requests) * 100
        
        # Performance benchmark: Error rate <1%
        self.assertLess(error_rate, 1.0, f"Error rate {error_rate}% exceeds 1% limit") 