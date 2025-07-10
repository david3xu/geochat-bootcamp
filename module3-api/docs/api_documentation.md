# Chat2MapMetadata API Documentation
## Full Stack AI Engineer Bootcamp - Module 3

---

## 📚 **API Overview**

The Chat2MapMetadata API provides real-time chat functionality with AI integration and spatial data services. Built with Django REST Framework and WebSocket support.

### **Base URL**
```
Development: http://localhost:8000
Production: https://chat2map-api.azurewebsites.net
```

### **API Version**
All endpoints are prefixed with `/api/v1/`

---

## 🔐 **Authentication**

The API uses JWT (JSON Web Token) authentication.

### **Obtain Token**
```http
POST /api/v1/auth/login/
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}
```

### **Response**
```json
{
    "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### **Using Token**
```http
Authorization: Bearer <your_access_token>
```

---

## 💬 **Chat Endpoints**

### **Chat Sessions**

#### **Create Session**
```http
POST /api/v1/chat/sessions/
Authorization: Bearer <token>
```

**Response:**
```json
{
    "id": "uuid",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "is_active": true,
    "session_metadata": {},
    "message_count": 0,
    "last_message_time": null
}
```

#### **List Sessions**
```http
GET /api/v1/chat/sessions/
Authorization: Bearer <token>
```

#### **End Session**
```http
POST /api/v1/chat/sessions/{session_id}/end_session/
Authorization: Bearer <token>
```

### **Chat Messages**

#### **Get Session Messages**
```http
GET /api/v1/chat/messages/session_messages/?session_id={session_id}
Authorization: Bearer <token>
```

**Response:**
```json
[
    {
        "id": "uuid",
        "message_type": "user",
        "content": "Hello AI",
        "timestamp": "2024-01-15T10:30:00Z",
        "processing_time": null,
        "ai_model_used": null,
        "spatial_context": {},
        "metadata": {}
    }
]
```

#### **Search Messages**
```http
GET /api/v1/chat/messages/search_messages/?q={search_query}
Authorization: Bearer <token>
```

**Response:**
```json
{
    "query": "search_query",
    "results": [...],
    "count": 5
}
```

### **Analytics**

#### **Response Time Analytics**
```http
GET /api/v1/chat/analytics/response_time_analytics/
Authorization: Bearer <token>
```

**Response:**
```json
{
    "total_queries": 150,
    "average_response_time": 185.5,
    "fast_queries": 120,
    "slow_queries": 5,
    "period": {
        "start": "2024-01-08T10:30:00Z",
        "end": "2024-01-15T10:30:00Z"
    }
}
```

#### **Session Analytics**
```http
GET /api/v1/chat/analytics/session_analytics/
Authorization: Bearer <token>
```

---

## 🔌 **WebSocket API**

### **Connection**
```
ws://localhost:8001/ws/chat/{session_id}/
```

### **Message Types**

#### **User Message**
```json
{
    "type": "user_message",
    "message": "Hello AI",
    "include_spatial_context": true
}
```

#### **Typing Indicator**
```json
{
    "type": "typing_indicator",
    "is_typing": true
}
```

### **Response Types**

#### **Connection Established**
```json
{
    "type": "connection_established",
    "session_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z",
    "message": "Connected to Chat2MapMetadata AI Assistant"
}
```

#### **User Message Received**
```json
{
    "type": "user_message_received",
    "message_id": "uuid",
    "content": "Hello AI",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

#### **AI Response**
```json
{
    "type": "ai_response",
    "message_id": "uuid",
    "content": "Hello! How can I help you with geological data?",
    "timestamp": "2024-01-15T10:30:01Z",
    "processing_time": 1.2,
    "spatial_context": {
        "locations": ["Western Australia"],
        "minerals": ["Gold"],
        "total_records": 5
    },
    "ai_confidence": 0.85
}
```

#### **Error**
```json
{
    "type": "error",
    "message": "Error description",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 🏥 **Health Check Endpoints**

### **System Health**
```http
GET /api/v1/health/
```

**Response:**
```json
{
    "overall_status": "healthy",
    "checks": {
        "database": {
            "status": "healthy",
            "response_time": 0.05,
            "details": {
                "connected": true,
                "postgis_version": "3.3.2"
            }
        },
        "redis": {
            "status": "healthy",
            "response_time": 0.02,
            "details": {
                "cache_connected": true,
                "channel_layer_connected": true
            }
        },
        "websocket": {
            "status": "healthy",
            "response_time": 0.01,
            "details": {
                "available": true
            }
        }
    },
    "total_response_time": 0.08,
    "timestamp": 1705312200.123
}
```

### **Performance Metrics**
```http
GET /api/v1/health/metrics/?period=1h
```

### **Performance Analytics**
```http
GET /api/v1/health/performance/
```

---

## 🗺️ **Spatial Endpoints**

### **Spatial Records**
```http
GET /api/v1/spatial/records/
Authorization: Bearer <token>
```

### **Spatial Search**
```http
POST /api/v1/spatial/search/
Authorization: Bearer <token>
Content-Type: application/json

{
    "query": "gold mining in Western Australia",
    "limit": 20
}
```

---

## 📊 **Error Codes**

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input data |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |

---

## 🚀 **Rate Limits**

- **API Requests**: 1000 requests per hour per user
- **WebSocket Connections**: 25 concurrent connections per user
- **Search Queries**: 100 queries per hour per user

---

## 📝 **Examples**

### **Complete Chat Flow**

1. **Create Session**
```bash
curl -X POST http://localhost:8000/api/v1/chat/sessions/ \
  -H "Authorization: Bearer <token>"
```

2. **Connect WebSocket**
```javascript
const ws = new WebSocket('ws://localhost:8001/ws/chat/{session_id}/');
```

3. **Send Message**
```javascript
ws.send(JSON.stringify({
    type: 'user_message',
    message: 'Tell me about gold mining in Western Australia',
    include_spatial_context: true
}));
```

4. **Receive Response**
```javascript
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'ai_response') {
        console.log('AI Response:', data.content);
        console.log('Spatial Context:', data.spatial_context);
    }
};
```

---

## 🔧 **Development**

### **Running Locally**
```bash
# Start services
docker-compose up -d

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver

# Start WebSocket server
python -m uvicorn src.asgi:application --reload --port 8001
```

### **Testing**
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"
pytest -m integration
pytest -m performance
```

---

**API Documentation Complete**: This comprehensive documentation provides all necessary information for developers to integrate with the Chat2MapMetadata API service, including authentication, endpoints, WebSocket communication, and deployment instructions. 