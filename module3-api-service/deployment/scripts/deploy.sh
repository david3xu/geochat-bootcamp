#!/bin/bash
# Deployment script for Chat2MapMetadata API Service
# Full Stack AI Engineer Bootcamp - Module 3

set -e

echo "🚀 Starting deployment of Chat2MapMetadata API Service..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t chat2map-api:latest .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start services
echo "▶️ Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run migrations
echo "🗄️ Running database migrations..."
docker-compose exec api python manage.py migrate

# Create superuser if needed
echo "👤 Creating superuser..."
docker-compose exec api python manage.py createsuperuser --noinput || true

# Check health
echo "🏥 Checking service health..."
curl -f http://localhost:8000/api/v1/health/ || {
    echo "❌ Health check failed"
    exit 1
}

echo "✅ Deployment completed successfully!"
echo "🌐 API is available at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/api/docs/"
echo "🔌 WebSocket endpoint: ws://localhost:8001/ws/chat/{session_id}/" 