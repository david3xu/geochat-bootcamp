#!/bin/bash
# Database migration script
# Full Stack AI Engineer Bootcamp - Module 3

set -e

echo "🗄️ Running database migrations..."

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --noinput

echo "✅ Migrations completed successfully!" 