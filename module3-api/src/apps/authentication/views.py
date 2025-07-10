"""
Authentication views for user management
Full Stack AI Engineer Bootcamp - Module 3
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .serializers import UserSerializer

class LoginView(TokenObtainPairView):
    """
    Custom login view with enhanced response
    Learning Outcome: JWT authentication implementation
    """
    permission_classes = [AllowAny]
    
    def post(self, request, *args, **kwargs):
        """Handle login request"""
        try:
            response = super().post(request, *args, **kwargs)
            
            # Add user information to response
            if response.status_code == 200:
                user = authenticate(
                    username=request.data.get('username'),
                    password=request.data.get('password')
                )
                if user:
                    response.data['user'] = {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'first_name': user.first_name,
                        'last_name': user.last_name
                    }
            
            return response
            
        except Exception as e:
            return Response(
                {'error': 'Login failed', 'details': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

class LogoutView(TokenRefreshView):
    """
    Logout view for token invalidation
    Learning Outcome: Token management and security
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        """Handle logout request"""
        try:
            # In a real implementation, you would blacklist the token
            return Response(
                {'message': 'Successfully logged out'},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(
                {'error': 'Logout failed', 'details': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

class UserViewSet(viewsets.ModelViewSet):
    """
    User management API
    Learning Outcome: User CRUD operations with authentication
    """
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter users based on permissions"""
        if self.request.user.is_staff:
            return User.objects.all()
        return User.objects.filter(id=self.request.user.id)
    
    @action(detail=False, methods=['get'])
    def profile(self, request):
        """Get current user profile"""
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)
    
    @action(detail=False, methods=['put'])
    def update_profile(self, request):
        """Update current user profile"""
        serializer = self.get_serializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
