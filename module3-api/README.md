# Module 3: API Service & Integration
## Full Stack AI Engineer Bootcamp with Snowflake Cortex Integration

---

## 🎯 **Learning Outcomes**

### **Technical Competencies**
- **Django REST Framework**: Professional API architecture with comprehensive endpoints
- **WebSocket Implementation**: Real-time bidirectional communication for chat functionality
- **Multi-Module Integration**: Seamless coordination between Module 1 (Data Pipeline) and Module 2 (AI Intelligence)
- **Authentication & Security**: JWT-based user management with role-based access control
- **Performance Monitoring**: Comprehensive health check system with real-time metrics

### **Measurable Success Criteria**
- **API Response Time**: <200ms average for chat endpoints
- **Concurrent Users**: Support 25+ simultaneous WebSocket connections
- **Integration Success**: 100% successful Module 1 & 2 communication
- **System Uptime**: 99.9% availability during load testing
- **Error Rate**: <1% request failure with graceful degradation

---

## 🏗️ **Project Architecture**

### **Core Components**
```
module3-api-service/
├── src/                    # Django project core
│   ├── settings/          # Environment-specific settings
│   ├── urls.py           # Main URL configuration
│   ├── asgi.py           # WebSocket support
│   └── wsgi.py           # WSGI application
├── apps/                  # Django applications
│   ├── chat/             # Real-time messaging
│   ├── spatial/          # Geospatial data services
│   ├── authentication/   # User management
│   ├── integration/      # Module communication
│   └── monitoring/       # Health & performance
├── tests/                # Comprehensive test suite
├── deployment/           # Production deployment configs
└── docs/                # API documentation
```

### **Technology Stack**
- **Backend**: Django 4.2.7 + Django REST Framework
- **Database**: PostgreSQL with PostGIS extension
- **Cache/Messaging**: Redis for WebSocket channels
- **WebSocket**: Django Channels with ASGI
- **AI Integration**: Snowflake Cortex client
- **Authentication**: JWT with SimpleJWT
- **Monitoring**: Custom health checks + metrics
- **Deployment**: Docker + Azure Container Apps

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL with PostGIS
- Redis

### **Local Development Setup**

```bash
# 1. Clone repository
git clone <repository-url>
cd module3-api-service

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp env.example .env
# Edit .env with your configuration

# 5. Start services
docker-compose up -d db redis

# 6. Run migrations
python manage.py migrate

# 7. Create superuser
python manage.py createsuperuser

# 8. Start development server
python manage.py runserver

# 9. Start WebSocket server (new terminal)
python -m uvicorn src.asgi:application --reload --port 8001
```

### **Production Deployment**

```bash
# 1. Build production image
docker build -t chat2map-api:latest .

# 2. Deploy to Azure Container Apps
az containerapp create \
  --name chat2map-api \
  --resource-group rg-chat2map \
  --image chat2map-api:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 10
```

---

## 📊 **API Endpoints**

### **Authentication**
- `POST /api/v1/auth/login/` - User login
- `POST /api/v1/auth/refresh/` - Token refresh
- `POST /api/v1/auth/logout/` - User logout

### **Chat Services**
- `GET/POST /api/v1/chat/sessions/` - Chat session management
- `GET /api/v1/chat/messages/` - Message retrieval
- `GET /api/v1/chat/analytics/` - Performance analytics

### **Spatial Services**
- `GET /api/v1/spatial/records/` - Spatial data records
- `POST /api/v1/spatial/search/` - Spatial search

### **Health & Monitoring**
- `GET /api/v1/health/` - System health status
- `GET /api/v1/health/metrics/` - Performance metrics
- `GET /api/v1/health/performance/` - Detailed analytics

### **WebSocket**
- `ws://localhost:8001/ws/chat/{session_id}/` - Real-time chat

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Django Settings
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database
DB_NAME=chat2map
DB_USER=postgres
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=5432

# Redis
REDIS_URL=redis://localhost:6379

# Integration Services
MODULE1_API_URL=http://localhost:8001
MODULE2_AI_URL=http://localhost:8002
SNOWFLAKE_CORTEX_URL=https://your-cortex-url.com
```

### **Docker Compose Services**
- **PostgreSQL**: Database with PostGIS
- **Redis**: Cache and WebSocket channels
- **API**: Main Django application
- **WebSocket**: ASGI server for real-time communication
- **Celery**: Background task processing

---

## 🧪 **Testing**

### **Run All Tests**
```bash
pytest
```

### **Test Categories**
```bash
# Unit tests
pytest tests/test_chat_api.py

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# WebSocket tests
pytest -m websocket

# Exclude slow tests
pytest -m "not slow"
```

### **Test Coverage**
```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# Coverage target: 80% minimum
pytest --cov=src --cov-fail-under=80
```

---

## 📈 **Performance Benchmarks**

### **API Performance**
- **Session Creation**: <100ms response time
- **Message Retrieval**: <200ms response time
- **Search Operations**: <300ms response time
- **Health Checks**: <1 second total time

### **WebSocket Performance**
- **Connection Time**: <100ms establishment
- **Message Latency**: <50ms delivery time
- **Concurrent Users**: 25+ simultaneous connections
- **Error Rate**: <1% message delivery failure

### **Integration Performance**
- **Module 1 Communication**: <500ms data retrieval
- **Module 2 AI Processing**: <2 seconds response time
- **Response Aggregation**: <50ms processing time
- **End-to-End Flow**: Complete user journey operational

---

## 🔍 **Monitoring & Health Checks**

### **System Health**
```bash
# Check overall system health
curl http://localhost:8000/api/v1/health/

# Get performance metrics
curl http://localhost:8000/api/v1/health/metrics/

# View detailed analytics
curl http://localhost:8000/api/v1/health/performance/
```

### **Health Check Components**
- **Database**: PostgreSQL connectivity and PostGIS functionality
- **Redis**: Cache and WebSocket channel layer
- **WebSocket**: Real-time communication availability
- **Module 1**: Spatial data service integration
- **Module 2**: AI service integration
- **External Services**: Snowflake Cortex availability

---

## 🚨 **Error Handling**

### **Graceful Degradation**
- **AI Service Unavailable**: Fallback responses with error notification
- **Spatial Service Unavailable**: Continue with AI-only responses
- **Database Issues**: Cached responses and error logging
- **WebSocket Failures**: Automatic reconnection with exponential backoff

### **Error Response Format**
```json
{
    "error": "Error description",
    "code": "ERROR_CODE",
    "timestamp": "2024-01-15T10:30:00Z",
    "details": {
        "service": "affected_service",
        "suggestion": "recovery_action"
    }
}
```

---

## 🔐 **Security Features**

### **Authentication**
- JWT-based authentication with refresh tokens
- Token expiration and rotation
- Secure password validation

### **Authorization**
- Role-based access control
- Session-based permissions
- API rate limiting

### **Data Protection**
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

---

## 📚 **Documentation**

### **API Documentation**
- **Swagger UI**: `http://localhost:8000/api/docs/`
- **ReDoc**: `http://localhost:8000/api/redoc/`
- **OpenAPI Schema**: `http://localhost:8000/api/schema/`

### **Integration Guides**
- [Module 1 Integration Guide](docs/integration_guide.md)
- [Module 2 AI Integration](docs/ai_integration.md)
- [WebSocket Implementation](docs/websocket_guide.md)

### **Deployment Guides**
- [Local Development](docs/local_development.md)
- [Production Deployment](docs/production_deployment.md)
- [Azure Container Apps](docs/azure_deployment.md)

---

## 🎓 **Learning Assessment**

### **Daily Checkpoints**

#### **Day 1: Django REST Foundation**
- [ ] Django project runs without errors
- [ ] All models migrate successfully
- [ ] Basic API endpoints respond correctly
- [ ] WebSocket connection established

#### **Day 2: Real-time Chat Implementation**
- [ ] WebSocket chat functionality operational
- [ ] Message persistence in database
- [ ] Real-time message delivery <50ms
- [ ] Error handling for connection failures

#### **Day 3: Multi-Module Integration**
- [ ] Module 1 integration successful
- [ ] Module 2 AI service integration functional
- [ ] Response aggregation working correctly
- [ ] End-to-end data flow operational

#### **Day 4: Performance & Security**
- [ ] Performance monitoring operational
- [ ] Security measures implemented
- [ ] Load testing passes (25+ concurrent users)
- [ ] Error rates <1%

#### **Day 5: Production Deployment**
- [ ] Production deployment successful
- [ ] All services healthy
- [ ] Performance targets met
- [ ] System monitoring operational

### **Assessment Commands**
```bash
# Day 1 Assessment
python manage.py check
python manage.py test apps.chat.tests.test_models
curl -X GET http://localhost:8000/api/v1/health/

# Day 2 Assessment
python manage.py test apps.chat.tests.test_websocket
python manage.py test apps.chat.tests.test_consumers

# Day 3 Assessment
python manage.py test apps.integration.tests.test_ai_client
python manage.py test apps.integration.tests.test_data_client

# Day 4 Assessment
python manage.py test apps.monitoring.tests.test_performance
curl -X GET http://localhost:8000/api/v1/health/metrics

# Day 5 Assessment
docker-compose up -d
docker-compose ps
curl -X GET http://localhost:8000/api/v1/health/
```

---

## 🤝 **Contributing**

### **Development Workflow**
1. Create feature branch from `main`
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation
5. Submit pull request

### **Code Standards**
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Maintain 80%+ test coverage
- Use type hints where appropriate

---

## 📄 **License**

This project is part of the Full Stack AI Engineer Bootcamp curriculum. All rights reserved.

---

**Module 3 Implementation Complete**: This comprehensive API service provides students with professional-grade experience in Django REST Framework, WebSocket real-time communication, multi-service integration, performance monitoring, and production deployment. The implementation includes measurable learning outcomes, comprehensive testing, and production-ready deployment configurations. 