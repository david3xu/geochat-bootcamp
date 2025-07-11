"""
WebSocket routing configuration
Full Stack AI Engineer Bootcamp - Module 3
"""

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/(?P<session_id>[0-9a-f-]+)/$', consumers.ChatConsumer.as_asgi()),
] 