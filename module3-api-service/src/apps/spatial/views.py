"""
Spatial data views
Full Stack AI Engineer Bootcamp - Module 3
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import SpatialDataSerializer

class SpatialDataViewSet(viewsets.ViewSet):
    """
    Spatial data API endpoints
    Learning Outcome: Geospatial data API design
    """
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """
        Search spatial data
        Learning Outcome: Spatial data search implementation
        """
        query = request.query_params.get('q', '')
        if not query:
            return Response(
                {'error': 'Query parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Mock spatial data response
        spatial_data = {
            'query': query,
            'results': [
                {
                    'id': 'geo_001',
                    'name': 'Gold Mine Site A',
                    'location': {'lat': -31.9505, 'lng': 115.8605},
                    'mineral_type': 'Gold',
                    'description': 'Active gold mining operation in Western Australia',
                    'confidence': 0.95
                },
                {
                    'id': 'geo_002',
                    'name': 'Iron Ore Deposit B',
                    'location': {'lat': -22.9068, 'lng': 114.2353},
                    'mineral_type': 'Iron Ore',
                    'description': 'Large iron ore deposit in Pilbara region',
                    'confidence': 0.88
                }
            ],
            'total_count': 2
        }
        
        return Response(spatial_data)
    
    @action(detail=False, methods=['get'])
    def nearby(self, request):
        """
        Find nearby spatial data
        Learning Outcome: Proximity-based spatial queries
        """
        lat = request.query_params.get('lat')
        lng = request.query_params.get('lng')
        radius = request.query_params.get('radius', 10)  # km
        
        if not lat or not lng:
            return Response(
                {'error': 'Latitude and longitude parameters are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Mock nearby data response
        nearby_data = {
            'center': {'lat': float(lat), 'lng': float(lng)},
            'radius': float(radius),
            'results': [
                {
                    'id': 'geo_003',
                    'name': 'Copper Mine C',
                    'location': {'lat': float(lat) + 0.01, 'lng': float(lng) + 0.01},
                    'mineral_type': 'Copper',
                    'distance': 1.2,  # km
                    'description': 'Copper mining operation'
                }
            ]
        }
        
        return Response(nearby_data)
