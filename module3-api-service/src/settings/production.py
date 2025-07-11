"""
Production settings for Chat2MapMetadata API Service
"""

from .base import *

# Security settings for production
DEBUG = False
ALLOWED_HOSTS = [
    'chat2map-api.azurewebsites.net',
    'api.chat2map.com',
]

# Security enhancements
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'

# Database configuration for production
DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': os.environ.get('AZURE_POSTGRESQL_NAME'),
        'USER': os.environ.get('AZURE_POSTGRESQL_USER'),
        'PASSWORD': os.environ.get('AZURE_POSTGRESQL_PASSWORD'),
        'HOST': os.environ.get('AZURE_POSTGRESQL_HOST'),
        'PORT': os.environ.get('AZURE_POSTGRESQL_PORT', '5432'),
        'OPTIONS': {
            'sslmode': 'require',
        },
    }
}

# Redis configuration for production
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [os.environ.get('AZURE_REDIS_URL')],
        },
    },
}

# Static files configuration for production
STATIC_URL = '/static/'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')

# Production logging
LOGGING['handlers']['file']['filename'] = '/var/log/chat2map/api.log' 