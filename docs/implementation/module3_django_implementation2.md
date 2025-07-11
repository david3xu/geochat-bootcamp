# Module 3: Django Implementation Guide
## Built-in Commands to Custom Architecture - Code Priority Approach

---

## 🎯 **Implementation Strategy: Django Built-in → Custom Professional**

**Week 3 Day 1-2**: Django foundation with built-in commands  
**Week 3 Day 3-5**: Custom architecture for Chat2MapMetadata integration

---

## 📋 **Phase 1: Django Foundation Setup (Day 1-2)**

### **Step 1: Create Django Project Structure**

```bash
# Navigate to your project root
cd Chat2MapMetadata

# Create Module 3 directory
mkdir module3-api-service
cd module3-api-service

# Create Django project using built-in command
django-admin startproject chat2map_api .

# Verify project structure
ls -la
# Output should show:
# manage.py
# chat2map_api/
#   ├── __init__.py
#   ├── settings.py
#   ├── urls.py
#   ├── wsgi.py
#   └── asgi.py
```

### **Step 2: Create Custom Apps for Our Architecture**

```bash
# Create our specialized apps
python manage.py startapp chat
python manage.py startapp spatial  
python manage.py startapp authentication
python manage.py startapp monitoring

# Verify app structure
ls -la
# Output should show:
# chat/
# spatial/
# authentication/
# monitoring/
# chat2map_api/
# manage.py
```

### **Step 3: Initial Django Configuration**

**File: `chat2map_api/settings.py`**
```python
"""
Django settings for Chat2MapMetadata API Service
Code-first approach with progressive complexity
"""
import os
from pathlib import Path
from datetime import timedelta

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-dev-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# Application definition - Our custom apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',  # For spatial data support
    
    # Third-party apps
    'rest_framework',
    'rest_framework_simplejwt',
    'channels',  # For WebSocket support
    'corsheaders',
    
    # Our custom apps
    'chat',
    'spatial',
    'authentication',
    'monitoring',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'chat2map_api.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI and ASGI applications
WSGI_APPLICATION = 'chat2map_api.wsgi.application'
ASGI_APPLICATION = 'chat2map_api.asgi.application'

# Database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': os.environ.get('DB_NAME', 'chat2map_db'),
        'USER': os.environ.get('DB_USER', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD', 'password'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

# Django REST Framework configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
}

# JWT Configuration
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
}

# Channels configuration for WebSocket
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}

# CORS settings
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Next.js frontend
    "http://127.0.0.1:3000",
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
```

### **Step 4: Configure ASGI for WebSocket Support**

**File: `chat2map_api/asgi.py`**
```python
"""
ASGI config for Chat2MapMetadata API Service
Enables WebSocket support for real-time chat
"""
import os
import django
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chat2map_api.settings')
django.setup()

# Import after Django setup
from chat.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        )
    ),
})
```

### **Step 5: Main URL Configuration**

**File: `chat2map_api/urls.py`**
```python
"""
Main URL configuration for Chat2MapMetadata API Service
Code-first routing with progressive complexity
"""
from django.contrib import admin
from django.urls import path, include

# API version
API_VERSION = 'v1'

urlpatterns = [
    # Admin interface
    path('admin/', admin.site.urls),
    
    # API endpoints
    path(f'api/{API_VERSION}/auth/', include('authentication.urls')),
    path(f'api/{API_VERSION}/chat/', include('chat.urls')),
    path(f'api/{API_VERSION}/spatial/', include('spatial.urls')),
    path(f'api/{API_VERSION}/health/', include('monitoring.urls')),
]
```

---

## 🔨 **Phase 2: Custom App Implementation (Day 3-5)**

### **App 1: Authentication Service**

**File: `authentication/models.py`**
```python
"""
Authentication models for Chat2MapMetadata
Simple user management with role-based access
"""
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    """Custom user model with additional fields"""
    role = models.CharField(
        max_length=20,
        choices=[
            ('admin', 'Administrator'),
            ('analyst', 'Data Analyst'), 
            ('viewer', 'Viewer'),
        ],
        default='viewer'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.username} ({self.role})"
```

**File: `authentication/serializers.py`**
```python
"""
Authentication serializers for API responses
Clean data validation and formatting
"""
from rest_framework import serializers
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .models import User

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm', 'role')

    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return data

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user

class UserLoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, data):
        username = data.get('username')
        password = data.get('password')
        
        if username and password:
            user = authenticate(username=username, password=password)
            if user:
                if user.is_active:
                    data['user'] = user
                else:
                    raise serializers.ValidationError("User account is disabled.")
            else:
                raise serializers.ValidationError("Invalid username or password.")
        else:
            raise serializers.ValidationError("Must provide username and password.")
        
        return data
```

**File: `authentication/views.py`**
```python
"""
Authentication views for Chat2MapMetadata API
Simple authentication with JWT tokens
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserRegistrationSerializer, UserLoginSerializer

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """User registration endpoint"""
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """User login endpoint"""
    serializer = UserLoginSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)
        return Response({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        })
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

**File: `authentication/urls.py`**
```python
"""
Authentication URL patterns
Simple and clean routing
"""
from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
```

### **App 2: Chat Service**

**File: `chat/models.py`**
```python
"""
Chat models for real-time messaging
Simple conversation and message tracking
"""
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Conversation(models.Model):
    """Chat conversation model"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Conversation {self.id} - {self.user.username}"

class Message(models.Model):
    """Individual message model"""
    conversation = models.ForeignKey(
        Conversation, 
        related_name='messages',
        on_delete=models.CASCADE
    )
    content = models.TextField()
    is_user_message = models.BooleanField(default=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # AI response metadata
    ai_confidence = models.FloatField(null=True, blank=True)
    processing_time = models.FloatField(null=True, blank=True)  # in seconds
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        sender = "User" if self.is_user_message else "AI"
        return f"{sender}: {self.content[:50]}..."
```

**File: `chat/consumers.py`**
```python
"""
WebSocket consumer for real-time chat
Handles WebSocket connections and message routing
"""
import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from .models import Conversation, Message

User = get_user_model()

class ChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for chat functionality"""
    
    async def connect(self):
        """Handle WebSocket connection"""
        self.conversation_id = self.scope['url_route']['kwargs']['conversation_id']
        self.room_group_name = f'chat_{self.conversation_id}'
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """Handle received WebSocket message"""
        text_data_json = json.loads(text_data)
        message_content = text_data_json['message']
        
        # Save user message to database
        message = await self.save_message(
            conversation_id=self.conversation_id,
            content=message_content,
            is_user_message=True
        )
        
        # Send user message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': {
                    'id': message.id,
                    'content': message_content,
                    'is_user_message': True,
                    'timestamp': message.timestamp.isoformat(),
                }
            }
        )
        
        # Process AI response (simulate for now)
        ai_response = await self.get_ai_response(message_content)
        
        # Save AI response to database
        ai_message = await self.save_message(
            conversation_id=self.conversation_id,
            content=ai_response,
            is_user_message=False
        )
        
        # Send AI response to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': {
                    'id': ai_message.id,
                    'content': ai_response,
                    'is_user_message': False,
                    'timestamp': ai_message.timestamp.isoformat(),
                }
            }
        )

    async def chat_message(self, event):
        """Send message to WebSocket"""
        message = event['message']
        
        await self.send(text_data=json.dumps({
            'type': 'message',
            'message': message
        }))

    @database_sync_to_async
    def save_message(self, conversation_id, content, is_user_message):
        """Save message to database"""
        conversation = Conversation.objects.get(id=conversation_id)
        return Message.objects.create(
            conversation=conversation,
            content=content,
            is_user_message=is_user_message
        )

    async def get_ai_response(self, user_message):
        """Get AI response (placeholder for Module 2 integration)"""
        # TODO: Integrate with Module 2 AI service
        await asyncio.sleep(1)  # Simulate processing time
        return f"AI Response to: {user_message}"
```

**File: `chat/routing.py`**
```python
"""
WebSocket URL routing for chat
"""
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/(?P<conversation_id>\w+)/$', consumers.ChatConsumer.as_asgi()),
]
```

**File: `chat/serializers.py`**
```python
"""
Chat serializers for API responses
"""
from rest_framework import serializers
from .models import Conversation, Message

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'content', 'is_user_message', 'timestamp', 'ai_confidence']

class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'is_active', 'messages', 'message_count']
    
    def get_message_count(self, obj):
        return obj.messages.count()
```

**File: `chat/views.py`**
```python
"""
Chat API views
RESTful endpoints for conversation management
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer

class ConversationViewSet(viewsets.ModelViewSet):
    """API endpoints for conversation management"""
    serializer_class = ConversationSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get messages for a specific conversation"""
        conversation = self.get_object()
        messages = conversation.messages.all()
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent conversations"""
        conversations = self.get_queryset()[:5]
        serializer = self.get_serializer(conversations, many=True)
        return Response(serializer.data)
```

**File: `chat/urls.py`**
```python
"""
Chat URL patterns
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'conversations', views.ConversationViewSet, basename='conversation')

urlpatterns = [
    path('', include(router.urls)),
]
```

### **App 3: Spatial Service**

**File: `spatial/models.py`**
```python
"""
Spatial models for geological data
Integration with Module 1 data pipeline
"""
from django.contrib.gis.db import models

class GeologicalSite(models.Model):
    """Geological site model with spatial data"""
    wamex_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    description = models.TextField()
    
    # Spatial fields
    location = models.PointField()
    area = models.PolygonField(null=True, blank=True)
    
    # Geological data
    mineral_type = models.CharField(max_length=100)
    depth = models.FloatField(null=True, blank=True)  # in meters
    estimated_reserves = models.FloatField(null=True, blank=True)
    
    # Metadata
    survey_date = models.DateField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('inactive', 'Inactive'),
            ('pending', 'Pending Review'),
        ],
        default='active'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.mineral_type})"
```

**File: `spatial/views.py`**
```python
"""
Spatial API views
Geospatial data access and search
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import Distance
from .models import GeologicalSite
from .serializers import GeologicalSiteSerializer

class SpatialDataViewSet(viewsets.ReadOnlyModelViewSet):
    """Spatial data API endpoints"""
    queryset = GeologicalSite.objects.all()
    serializer_class = GeologicalSiteSerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Search geological sites by text"""
        query = request.query_params.get('q', '')
        mineral_type = request.query_params.get('mineral_type', '')
        
        queryset = self.get_queryset()
        
        if query:
            queryset = queryset.filter(
                models.Q(name__icontains=query) |
                models.Q(description__icontains=query)
            )
        
        if mineral_type:
            queryset = queryset.filter(mineral_type__icontains=mineral_type)
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'query': query,
            'mineral_type': mineral_type,
            'count': queryset.count(),
            'results': serializer.data
        })
    
    @action(detail=False, methods=['get'])
    def nearby(self, request):
        """Find sites near a location"""
        lat = request.query_params.get('lat')
        lng = request.query_params.get('lng')
        radius = float(request.query_params.get('radius', 10))  # km
        
        if not lat or not lng:
            return Response(
                {'error': 'Latitude and longitude parameters are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        point = Point(float(lng), float(lat))
        queryset = self.get_queryset().filter(
            location__distance_lte=(point, Distance(km=radius))
        )
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'center': {'lat': float(lat), 'lng': float(lng)},
            'radius_km': radius,
            'count': queryset.count(),
            'results': serializer.data
        })
    
    @action(detail=False, methods=['get'])
    def minerals(self, request):
        """Get list of available mineral types"""
        mineral_types = GeologicalSite.objects.values_list(
            'mineral_type', flat=True
        ).distinct().order_by('mineral_type')
        
        return Response({
            'mineral_types': list(mineral_types)
        })
```

**File: `spatial/serializers.py`**
```python
"""
Spatial data serializers
GeoJSON-compatible serialization
"""
from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from .models import GeologicalSite

class GeologicalSiteSerializer(GeoFeatureModelSerializer):
    """GeoJSON serializer for geological sites"""
    
    class Meta:
        model = GeologicalSite
        geo_field = 'location'
        fields = [
            'id', 'wamex_id', 'name', 'description', 'mineral_type',
            'depth', 'estimated_reserves', 'survey_date', 'status',
            'created_at', 'updated_at'
        ]
```

**File: `spatial/urls.py`**
```python
"""
Spatial URL patterns
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'sites', views.SpatialDataViewSet, basename='spatial')

urlpatterns = [
    path('', include(router.urls)),
]
```

### **App 4: Monitoring Service**

**File: `monitoring/views.py`**
```python
"""
Monitoring and health check views
System health and performance metrics
"""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.db import connection
from django.core.cache import cache
import time

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """System health check endpoint"""
    start_time = time.time()
    
    # Check database connection
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        db_status = "healthy"
        db_response_time = (time.time() - start_time) * 1000
    except Exception as e:
        db_status = f"error: {str(e)}"
        db_response_time = None
    
    # Check cache
    try:
        cache.set('health_check', 'test', 30)
        cache_value = cache.get('health_check')
        cache_status = "healthy" if cache_value == 'test' else "error"
    except Exception as e:
        cache_status = f"error: {str(e)}"
    
    total_response_time = (time.time() - start_time) * 1000
    
    return Response({
        'status': 'healthy',
        'timestamp': time.time(),
        'services': {
            'database': {
                'status': db_status,
                'response_time_ms': db_response_time
            },
            'cache': {
                'status': cache_status
            }
        },
        'response_time_ms': total_response_time,
        'version': '1.0.0'
    })

@api_view(['GET'])
def metrics(request):
    """System metrics endpoint"""
    from django.db import models
    from chat.models import Conversation, Message
    from spatial.models import GeologicalSite
    
    return Response({
        'conversations': {
            'total': Conversation.objects.count(),
            'active': Conversation.objects.filter(is_active=True).count(),
        },
        'messages': {
            'total': Message.objects.count(),
            'user_messages': Message.objects.filter(is_user_message=True).count(),
            'ai_messages': Message.objects.filter(is_user_message=False).count(),
        },
        'spatial_data': {
            'total_sites': GeologicalSite.objects.count(),
            'active_sites': GeologicalSite.objects.filter(status='active').count(),
        }
    })
```

**File: `monitoring/urls.py`**
```python
"""
Monitoring URL patterns
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.health_check, name='health_check'),
    path('metrics/', views.metrics, name='metrics'),
]
```

---

## 🚀 **Phase 3: Deployment Configuration**

### **Requirements File**

**File: `requirements.txt`**
```txt
# Django core
Django==4.2.7
djangorestframework==3.14.0
django-cors-headers==4.3.1

# Authentication
djangorestframework-simplejwt==5.3.0

# Spatial support
django-gis==0.24

# WebSocket support
channels==4.0.0
channels-redis==4.1.0

# Database
psycopg2-binary==2.9.7

# Development tools
django-debug-toolbar==4.2.0

# Production server
gunicorn==21.2.0
uvicorn[standard]==0.24.0
```

### **Docker Configuration**

**File: `Dockerfile`**
```dockerfile
# Module 3: API Service Container
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    gdal-bin \
    libgdal-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app
ENV DJANGO_SETTINGS_MODULE=chat2map_api.settings

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "chat2map_api.wsgi:application"]
```

### **Docker Compose**

**File: `docker-compose.yml`**
```yaml
version: '3.8'

services:
  # Redis for WebSocket channels
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # PostgreSQL with PostGIS
  db:
    image: postgis/postgis:15-3.3
    environment:
      POSTGRES_DB: chat2map_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Django API service
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - DB_NAME=chat2map_db
      - DB_USER=postgres
      - DB_PASSWORD=password
      - DEBUG=True
    depends_on:
      - db
      - redis
    volumes:
      - .:/app

volumes:
  postgres_data:
```

---

## 📋 **Complete Setup Commands**

### **Initial Setup**
```bash
# 1. Create project structure
mkdir module3-api-service && cd module3-api-service

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Django
pip install django djangorestframework django-cors-headers

# 4. Create Django project
django-admin startproject chat2map_api .

# 5. Create apps
python manage.py startapp chat
python manage.py startapp spatial
python manage.py startapp authentication
python manage.py startapp monitoring

# 6. Install additional dependencies
pip install djangorestframework-simplejwt channels channels-redis psycopg2-binary

# 7. Configure settings (copy the settings.py content above)

# 8. Run initial migrations
python manage.py makemigrations
python manage.py migrate

# 9. Create superuser
python manage.py createsuperuser

# 10. Run development server
python manage.py runserver
```

### **Development Workflow**
```bash
# Make migrations after model changes
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Run tests
python manage.py test

# Run development server
python manage.py runserver 0.0.0.0:8000
```

### **Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run migrations in container
docker-compose exec api python manage.py migrate

# Create superuser in container
docker-compose exec api python manage.py createsuperuser

# View logs
docker-compose logs -f api
```

---

## ✅ **Verification Commands**

### **Test API Endpoints**
```bash
# Health check
curl http://localhost:8000/api/v1/health/

# Register user
curl -X POST http://localhost:8000/api/v1/auth/register/ \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "email": "test@example.com", "password": "testpass123", "password_confirm": "testpass123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "testpass123"}'

# Get conversations (requires JWT token)
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:8000/api/v1/chat/conversations/

# Search spatial data
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  "http://localhost:8000/api/v1/spatial/sites/search/?q=gold"
```

---

## 🎯 **Success Metrics**

**Day 1-2 Completion:**
- ✅ Django project created with built-in commands
- ✅ Custom apps configured for Chat2MapMetadata architecture
- ✅ Basic authentication and health endpoints working

**Day 3-5 Completion:**
- ✅ WebSocket chat functionality operational
- ✅ Spatial data API with GeoJSON support
- ✅ JWT authentication system
- ✅ Production-ready Docker deployment

**Integration Ready:**
- ✅ Module 1 spatial data integration points
- ✅ Module 2 AI service integration hooks
- ✅ Module 4 frontend API contracts
- ✅ Real-time WebSocket communication

This implementation follows Django best practices while maintaining our code-first, simple-to-professional approach with clear progression from built-in commands to custom architecture.