"""
Chat application URL configuration
Full Stack AI Engineer Bootcamp - Module 3
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'sessions', views.ChatSessionViewSet, basename='chatsession')
router.register(r'messages', views.ChatMessageViewSet, basename='chatmessage')
router.register(r'analytics', views.PerformanceAnalyticsView, basename='analytics')

urlpatterns = [
    path('', include(router.urls)),
]
