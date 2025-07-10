"""
Chat API views for REST endpoints
Full Stack AI Engineer Bootcamp - Module 3
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count, Avg, Max
from django.utils import timezone
from datetime import timedelta
import logging

from .models import ChatSession, ChatMessage, UserQuery
from .serializers import (
    ChatSessionSerializer, ChatMessageSerializer, 
    UserQuerySerializer, ChatRequestSerializer
)
from apps.integration.ai_client import SnowflakeCortexClient
from apps.integration.data_client import SpatialDataClient
from apps.integration.response_aggregator import ResponseAggregator

logger = logging.getLogger('chat2map')

class ChatSessionViewSet(viewsets.ModelViewSet):
    """
    Chat session management API
    Learning Outcome: RESTful API design for real-time applications
    
    Measurable Success Criteria:
    - Create new chat sessions: <100ms response time
    - Retrieve session history: <200ms response time
    - Support 25+ concurrent sessions
    """
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter sessions by authenticated user"""
        return ChatSession.objects.filter(user=self.request.user).annotate(
            message_count=Count('messages'),
            last_message_time=Max('messages__timestamp')
        )
    
    def perform_create(self, serializer):
        """Create new chat session for authenticated user"""
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def end_session(self, request, pk=None):
        """
        End chat session
        Learning Outcome: Custom API actions for business logic
        """
        session = self.get_object()
        session.is_active = False
        session.save()
        
        return Response({
            'message': 'Session ended successfully',
            'session_id': session.id,
            'ended_at': timezone.now()
        })
    
    @action(detail=False, methods=['get'])
    def active_sessions(self, request):
        """
        Get active sessions for user
        Learning Outcome: Filtered API responses
        """
        active_sessions = self.get_queryset().filter(is_active=True)
        serializer = self.get_serializer(active_sessions, many=True)
        return Response(serializer.data)

class ChatMessageViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Chat message retrieval API
    Learning Outcome: Read-only API design for message history
    
    Measurable Success Criteria:
    - Message retrieval: <150ms response time
    - Paginated responses: 25 messages per page
    - Search functionality: <300ms response time
    """
    serializer_class = ChatMessageSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter messages by user's sessions"""
        user_sessions = ChatSession.objects.filter(user=self.request.user)
        return ChatMessage.objects.filter(session__in=user_sessions)
    
    @action(detail=False, methods=['get'])
    def session_messages(self, request):
        """
        Get messages for specific session
        Learning Outcome: Parameterized API queries
        """
        session_id = request.query_params.get('session_id')
        if not session_id:
            return Response(
                {'error': 'session_id parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response(
                {'error': 'Session not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        messages = self.get_queryset().filter(session=session)
        serializer = self.get_serializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def search_messages(self, request):
        """
        Search messages by content
        Learning Outcome: Full-text search implementation
        """
        query = request.query_params.get('q', '')
        if len(query) < 3:
            return Response(
                {'error': 'Search query must be at least 3 characters'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        messages = self.get_queryset().filter(content__icontains=query)
        serializer = self.get_serializer(messages, many=True)
        return Response({
            'query': query,
            'results': serializer.data,
            'count': messages.count()
        })

class PerformanceAnalyticsView(viewsets.ViewSet):
    """
    Performance analytics API
    Learning Outcome: Analytics and monitoring API design
    
    Measurable Success Criteria:
    - Analytics queries: <500ms response time
    - Real-time metrics: Update every 30 seconds
    - Performance trending: 7-day historical data
    """
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def response_time_analytics(self, request):
        """
        Get response time analytics
        Learning Outcome: Performance monitoring and reporting
        """
        end_date = timezone.now()
        start_date = end_date - timedelta(days=7)
        
        # Query performance metrics
        queries = UserQuery.objects.filter(
            session__user=request.user,
            timestamp__range=[start_date, end_date]
        )
        
        analytics = {
            'total_queries': queries.count(),
            'average_response_time': queries.aggregate(
                avg_time=Avg('response_time')
            )['avg_time'] or 0,
            'fast_queries': queries.filter(response_time__lt=200).count(),
            'slow_queries': queries.filter(response_time__gt=1000).count(),
            'period': {
                'start': start_date,
                'end': end_date
            }
        }
        
        return Response(analytics)
    
    @action(detail=False, methods=['get'])
    def session_analytics(self, request):
        """
        Get session analytics
        Learning Outcome: User behavior analytics
        """
        sessions = ChatSession.objects.filter(user=request.user)
        
        analytics = {
            'total_sessions': sessions.count(),
            'active_sessions': sessions.filter(is_active=True).count(),
            'average_messages_per_session': sessions.annotate(
                msg_count=Count('messages')
            ).aggregate(avg_messages=Avg('msg_count'))['avg_messages'] or 0,
            'total_messages': ChatMessage.objects.filter(
                session__user=request.user
            ).count()
        }
        
        return Response(analytics)
