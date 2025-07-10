from django.db import models
from django.contrib.auth.models import User
import uuid

class ChatSession(models.Model):
    """
    Django model for chat session management
    Measurable Success: Support 100+ concurrent sessions
    """
    
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        # Database indexes for performance optimization
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['is_active', '-last_activity']),
        ]

class ChatMessage(models.Model):
    """
    Individual chat message storage with AI response tracking
    Measurable Success: <50ms message storage and retrieval
    """
    
    message_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    content = models.TextField()
    message_type = models.CharField(max_length=20)  # 'user' or 'ai'
    timestamp = models.DateTimeField(auto_now_add=True)
    ai_processing_time = models.FloatField(null=True, blank=True)
    relevance_score = models.FloatField(null=True, blank=True)
    spatial_results = models.JSONField(null=True, blank=True)

class GeologicalQuery(models.Model):
    """
    Specialized storage for geological query analytics
    Measurable Success: 100% query pattern tracking for optimization
    """
    
    query_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    original_query = models.TextField()
    processed_query = models.TextField()
    query_type = models.CharField(max_length=50)
    spatial_bounds = models.JSONField(null=True, blank=True)
    mineral_types = models.JSONField(null=True, blank=True)
    response_time = models.FloatField()
    result_count = models.IntegerField()
