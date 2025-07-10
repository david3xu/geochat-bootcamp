#!/bin/bash
# Health check script
# Full Stack AI Engineer Bootcamp - Module 3

set -e

echo "🏥 Running health checks..."

# Check API health
echo "🔍 Checking API health..."
curl -f http://localhost:8000/api/v1/health/ || {
    echo "❌ API health check failed"
    exit 1
}

# Check WebSocket health
echo "🔌 Checking WebSocket health..."
curl -f http://localhost:8001/health/ || {
    echo "❌ WebSocket health check failed"
    exit 1
}

# Check database connection
echo "🗄️ Checking database connection..."
python manage.py check --database default || {
    echo "❌ Database health check failed"
    exit 1
}

# Check Redis connection
echo "🔴 Checking Redis connection..."
python -c "
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
r.ping()
print('Redis connection successful')
" || {
    echo "❌ Redis health check failed"
    exit 1
}

echo "✅ All health checks passed!" 