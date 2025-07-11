"""
Monitoring application URL configuration
Full Stack AI Engineer Bootcamp - Module 3
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'', views.MonitoringViewSet, basename='monitoring')

urlpatterns = [
    path('', include(router.urls)),
] 