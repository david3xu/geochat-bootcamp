"""
Spatial data serializers
Full Stack AI Engineer Bootcamp - Module 3
"""

from rest_framework import serializers

class SpatialDataSerializer(serializers.Serializer):
    """
    Spatial data serializer
    Learning Outcome: Geospatial data serialization
    """
    id = serializers.CharField()
    name = serializers.CharField()
    location = serializers.DictField()
    mineral_type = serializers.CharField()
    description = serializers.CharField()
    confidence = serializers.FloatField(required=False)
    distance = serializers.FloatField(required=False)
