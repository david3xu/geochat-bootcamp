"""
WebSocket consumers for real-time chat
Full Stack AI Engineer Bootcamp - Module 3
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from django.utils import timezone
from asgiref.sync import sync_to_async
import asyncio
import time

from .models import ChatSession, ChatMessage, UserQuery
from apps.integration.ai_client import SnowflakeCortexClient
from apps.integration.data_client import SpatialDataClient
from apps.integration.response_aggregator import ResponseAggregator

logger = logging.getLogger('chat2map')

class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time chat
    Learning Outcome: Real-time bidirectional communication
    
    Measurable Success Criteria:
    - WebSocket connection: <100ms establishment time
    - Message delivery: <50ms latency
    - Concurrent connections: Support 25+ users
    - Error handling: 99.9% message delivery success
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = None
        self.user = None
        self.ai_client = SnowflakeCortexClient()
        self.spatial_client = SpatialDataClient()
        self.response_aggregator = ResponseAggregator()
    
    async def connect(self):
        """
        Handle WebSocket connection
        Learning Outcome: WebSocket lifecycle management
        """
        # Extract session ID from URL
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.user = self.scope['user']
        
        # Validate user authentication
        if not self.user.is_authenticated:
            await self.close()
            return
        
        # Validate session ownership
        session_exists = await self.check_session_exists()
        if not session_exists:
            await self.close()
            return
        
        # Join session group
        self.session_group_name = f'chat_session_{self.session_id}'
        await self.channel_layer.group_add(
            self.session_group_name,
            self.channel_name
        )
        
        # Accept connection
        await self.accept()
        
        # Send connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'session_id': str(self.session_id),
            'timestamp': timezone.now().isoformat(),
            'message': 'Connected to Chat2MapMetadata AI Assistant'
        }))
        
        logger.info(f"WebSocket connected: user={self.user.username}, session={self.session_id}")
    
    async def disconnect(self, close_code):
        """
        Handle WebSocket disconnection
        Learning Outcome: Resource cleanup and connection management
        """
        # Leave session group
        if hasattr(self, 'session_group_name'):
            await self.channel_layer.group_discard(
                self.session_group_name,
                self.channel_name
            )
        
        logger.info(f"WebSocket disconnected: user={self.user.username}, session={self.session_id}")
    
    async def receive(self, text_data):
        """
        Handle incoming WebSocket messages
        Learning Outcome: Real-time message processing and AI integration
        """
        start_time = time.time()
        
        try:
            # Parse incoming message
            data = json.loads(text_data)
            message_type = data.get('type', 'user_message')
            
            if message_type == 'user_message':
                await self.handle_user_message(data, start_time)
            elif message_type == 'typing_indicator':
                await self.handle_typing_indicator(data)
            else:
                await self.send_error('Unknown message type')
                
        except json.JSONDecodeError:
            await self.send_error('Invalid JSON format')
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_error('Message processing failed')
    
    async def handle_user_message(self, data, start_time):
        """
        Process user message and generate AI response
        Learning Outcome: Multi-service integration and response orchestration
        """
        try:
            message_content = data.get('message', '').strip()
            include_spatial = data.get('include_spatial_context', True)
            
            if not message_content:
                await self.send_error('Message cannot be empty')
                return
            
            # Save user message
            user_message = await self.save_user_message(message_content)
            
            # Send user message confirmation
            await self.send(text_data=json.dumps({
                'type': 'user_message_received',
                'message_id': str(user_message.id),
                'content': message_content,
                'timestamp': user_message.timestamp.isoformat()
            }))
            
            # Send typing indicator
            await self.send(text_data=json.dumps({
                'type': 'ai_typing',
                'message': 'AI is thinking...'
            }))
            
            # Process with AI and spatial services
            ai_response = await self.process_with_ai(
                message_content, 
                include_spatial, 
                start_time
            )
            
            # Send AI response
            await self.send(text_data=json.dumps({
                'type': 'ai_response',
                'message_id': str(ai_response['message_id']),
                'content': ai_response['content'],
                'timestamp': ai_response['timestamp'],
                'processing_time': ai_response['processing_time'],
                'spatial_context': ai_response.get('spatial_context', {}),
                'ai_confidence': ai_response.get('ai_confidence', 0.0)
            }))
            
            # Record query metrics
            await self.record_query_metrics(
                message_content, 
                ai_response['processing_time'],
                ai_response.get('spatial_results_count', 0),
                ai_response.get('ai_confidence', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            await self.send_error('Failed to process message')
    
    async def process_with_ai(self, message_content, include_spatial, start_time):
        """
        Process message with AI and spatial services
        Learning Outcome: Asynchronous service integration
        """
        try:
            # Parallel processing of AI and spatial services
            ai_task = asyncio.create_task(
                self.ai_client.process_message(message_content)
            )
            
            spatial_task = None
            if include_spatial:
                spatial_task = asyncio.create_task(
                    self.spatial_client.search_relevant_data(message_content)
                )
            
            # Wait for AI response
            ai_result = await ai_task
            
            # Wait for spatial data if requested
            spatial_result = None
            if spatial_task:
                spatial_result = await spatial_task
            
            # Aggregate response
            aggregated_response = await self.response_aggregator.combine_responses(
                ai_result, 
                spatial_result
            )
            
            # Save AI response
            ai_message = await self.save_ai_message(
                aggregated_response['content'],
                time.time() - start_time,
                aggregated_response.get('spatial_context', {}),
                aggregated_response.get('ai_confidence', 0.0)
            )
            
            return {
                'message_id': ai_message.id,
                'content': aggregated_response['content'],
                'timestamp': ai_message.timestamp.isoformat(),
                'processing_time': ai_message.processing_time,
                'spatial_context': aggregated_response.get('spatial_context', {}),
                'spatial_results_count': aggregated_response.get('spatial_results_count', 0),
                'ai_confidence': aggregated_response.get('ai_confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error processing with AI: {e}")
            raise
    
    async def handle_typing_indicator(self, data):
        """
        Handle typing indicator messages
        Learning Outcome: Real-time user interaction feedback
        """
        is_typing = data.get('is_typing', False)
        
        # Broadcast typing indicator to session group
        await self.channel_layer.group_send(
            self.session_group_name,
            {
                'type': 'typing_indicator',
                'user': self.user.username,
                'is_typing': is_typing
            }
        )
    
    async def send_error(self, message):
        """Send error message to client"""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': message,
            'timestamp': timezone.now().isoformat()
        }))
    
    @database_sync_to_async
    def check_session_exists(self):
        """Check if session exists and belongs to user"""
        try:
            ChatSession.objects.get(id=self.session_id, user=self.user)
            return True
        except ChatSession.DoesNotExist:
            return False
    
    @database_sync_to_async
    def save_user_message(self, content):
        """Save user message to database"""
        session = ChatSession.objects.get(id=self.session_id)
        return ChatMessage.objects.create(
            session=session,
            message_type='user',
            content=content
        )
    
    @database_sync_to_async
    def save_ai_message(self, content, processing_time, spatial_context, ai_confidence):
        """Save AI message to database"""
        session = ChatSession.objects.get(id=self.session_id)
        return ChatMessage.objects.create(
            session=session,
            message_type='ai',
            content=content,
            processing_time=processing_time * 1000,  # Convert to milliseconds
            spatial_context=spatial_context,
            ai_model_used='snowflake-cortex',
            metadata={'ai_confidence': ai_confidence}
        )
    
    @database_sync_to_async
    def record_query_metrics(self, query_text, response_time, spatial_results_count, ai_confidence):
        """Record query metrics for analytics"""
        session = ChatSession.objects.get(id=self.session_id)
        UserQuery.objects.create(
            session=session,
            query_text=query_text,
            response_time=response_time * 1000,  # Convert to milliseconds
            spatial_results_count=spatial_results_count,
            ai_confidence_score=ai_confidence
        ) 