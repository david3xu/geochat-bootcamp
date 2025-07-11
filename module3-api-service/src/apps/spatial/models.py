"""
Spatial data models
Full Stack AI Engineer Bootcamp - Module 3
"""

from django.contrib.gis.db import models
from django.contrib.gis.geos import Point
import uuid

class SpatialRecord(models.Model):
    """
    Spatial record model for geological data
    Learning Outcome: Geospatial data modeling with PostGIS
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField()
    location = models.PointField(geography=True)
    mineral_type = models.CharField(max_length=100)
    confidence_score = models.FloatField(default=0.0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'spatial_records'
        indexes = [
            models.Index(fields=['mineral_type']),
            models.Index(fields=['confidence_score']),
        ]
    
    def __str__(self):
        return f"{self.name} - {self.mineral_type}"
    
    def set_location(self, lat, lng):
        """Set location from latitude and longitude"""
        self.location = Point(lng, lat, srid=4326)
    
    def get_lat_lng(self):
        """Get latitude and longitude from Point"""
        return (self.location.y, self.location.x)
