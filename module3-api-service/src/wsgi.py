"""
WSGI config for Chat2MapMetadata API Service
Full Stack AI Engineer Bootcamp - Module 3
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'src.settings.development')

application = get_wsgi_application() 