"""
Development settings for Chat2MapMetadata API Service
"""

from .base import *

# Security settings for development
DEBUG = True
ALLOWED_HOSTS = ['*']
SECRET_KEY = 'dev-secret-key-not-for-production'

# Database configuration for development
DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': 'chat2map_dev',
        'USER': 'postgres',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Disable security for development
CORS_ALLOW_ALL_ORIGINS = True

# Development-specific middleware
MIDDLEWARE += [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

INSTALLED_APPS += [
    'debug_toolbar',
]

# Debug toolbar configuration
INTERNAL_IPS = [
    '127.0.0.1',
]

# Mock services for development
USE_MOCK_SERVICES = True 