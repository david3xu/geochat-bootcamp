"""
Chat serializers for API responses
Full Stack AI Engineer Bootcamp - Module 3
"""

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import ChatSession, ChatMessage, UserQuery

class ChatSessionSerializer(serializers.ModelSerializer):
    """
    Chat session serializer
    Learning Outcome: API response design for real-time applications
    """
    message_count = serializers.IntegerField(read_only=True)
    last_message_time = serializers.DateTimeField(read_only=True)
    
    class Meta:
        model = ChatSession
        fields = [
            'id', 'created_at', 'updated_at', 'is_active',
            'session_metadata', 'message_count', 'last_message_time'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

class ChatMessageSerializer(serializers.ModelSerializer):
    """
    Chat message serializer with performance metrics
    Learning Outcome: Structured data representation for real-time messaging
    """
    class Meta:
        model = ChatMessage
        fields = [
            'id', 'message_type', 'content', 'timestamp',
            'processing_time', 'ai_model_used', 'spatial_context', 'metadata'
        ]
        read_only_fields = ['id', 'timestamp']

class UserQuerySerializer(serializers.ModelSerializer):
    """
    User query serializer with analytics
    Learning Outcome: Performance tracking and analytics implementation
    """
    class Meta:
        model = UserQuery
        fields = [
            'id', 'query_text', 'timestamp', 'response_time',
            'spatial_results_count', 'ai_confidence_score'
        ]
        read_only_fields = ['id', 'timestamp']

class ChatRequestSerializer(serializers.Serializer):
    """
    Chat request serializer for WebSocket messages
    Learning Outcome: Real-time data validation and processing
    """
    message = serializers.CharField(max_length=1000)
    session_id = serializers.UUIDField()
    include_spatial_context = serializers.BooleanField(default=True)
    
    def validate_message(self, value):
        """Validate message content"""
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()

class ChatResponseSerializer(serializers.Serializer):
    """
    Chat response serializer for WebSocket responses
    Learning Outcome: Structured real-time response formatting
    """
    message_id = serializers.UUIDField()
    content = serializers.CharField()
    message_type = serializers.CharField()
    timestamp = serializers.DateTimeField()
    processing_time = serializers.FloatField()
    spatial_context = serializers.JSONField()
    ai_confidence = serializers.FloatField()
