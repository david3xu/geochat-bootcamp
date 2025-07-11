"""
Chat models for real-time messaging
Full Stack AI Engineer Bootcamp - Module 3
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid

class ChatSession(models.Model):
    """
    Chat session model for tracking user conversations
    Learning Outcome: Database design for real-time applications
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    session_metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'chat_sessions'
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['is_active']),
        ]
    
    def __str__(self):
        return f"Session {self.id} - {self.user.username}"

class ChatMessage(models.Model):
    """
    Chat message model for storing conversation history
    Learning Outcome: Efficient message storage and retrieval
    """
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('ai', 'AI Response'),
        ('system', 'System Message'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    # Performance tracking fields
    processing_time = models.FloatField(null=True, blank=True)  # milliseconds
    ai_model_used = models.CharField(max_length=50, blank=True)
    spatial_context = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'chat_messages'
        indexes = [
            models.Index(fields=['session', 'timestamp']),
            models.Index(fields=['message_type']),
        ]
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."
    
    def save(self, *args, **kwargs):
        """Override save to update session timestamp"""
        super().save(*args, **kwargs)
        self.session.updated_at = timezone.now()
        self.session.save()

class UserQuery(models.Model):
    """
    User query model for tracking search patterns and performance
    Learning Outcome: Analytics and performance monitoring
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    query_text = models.TextField()
    query_embedding = models.JSONField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Performance metrics
    response_time = models.FloatField(null=True, blank=True)  # milliseconds
    spatial_results_count = models.IntegerField(default=0)
    ai_confidence_score = models.FloatField(null=True, blank=True)
    
    class Meta:
        db_table = 'user_queries'
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['session']),
        ]
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}..."
