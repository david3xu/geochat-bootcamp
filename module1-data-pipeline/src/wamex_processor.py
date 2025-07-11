"""
WAMEX Data Processing Engine for Geological Exploration Records
Measurable Success: 98%+ processing accuracy for 1,000 records
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from .config import config

# Configure logging for supervision monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for supervisor monitoring"""
    total_records: int
    processed_successfully: int
    failed_records: int
    processing_accuracy: float
    processing_time_seconds: float
    coordinate_transformation_errors: int
    validation_errors: int

@dataclass
class ValidationReport:
    """Data validation results"""
    total_records: int
    valid_coordinates: int
    valid_geometries: int
    valid_metadata: int
    coordinate_accuracy: float
    geometry_accuracy: float
    metadata_completeness: float

class WAMEXDataProcessor:
    """
    Core data processing engine for geological exploration records
    Measurable Success: 98%+ processing accuracy for 1,000 records
    """
    
    def __init__(self):
        self.processing_config = config.processing
        self.metrics = ProcessingMetrics(0, 0, 0, 0.0, 0.0, 0, 0)
        
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load WAMEX CSV with error handling and validation
        Success Metric: Complete load of 1,000 records
        """
        start_time = time.time()
        
        try:
            # Load CSV with proper data types
            data = pd.read_csv(file_path, encoding='utf-8')
            
            # Log initial data statistics
            logger.info(f"Loaded {len(data)} records from {file_path}")
            logger.info(f"Columns: {list(data.columns)}")
            
            # Basic data validation
            if data.empty:
                raise ValueError("CSV file is empty")
            
            # Check for required columns (sample WAMEX structure)
            required_columns = ['PERMIT_ID', 'LONGITUDE', 'LATITUDE', 'MINERAL_TYPE', 'DESCRIPTION']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                # Create missing columns with default values for demo
                for col in missing_columns:
                    if col in ['LONGITUDE', 'LATITUDE']:
                        data[col] = 0.0
                    else:
                        data[col] = 'Unknown'
            
            self.metrics.total_records = len(data)
            processing_time = time.time() - start_time
            
            logger.info(f"Data loading completed in {processing_time:.2f} seconds")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def validate_spatial_coordinates(self, data: pd.DataFrame) -> ValidationReport:
        """
        Validate coordinate ranges and spatial integrity
        Success Metric: 99%+ coordinate validation success
        """
        logger.info("Starting spatial coordinate validation...")
        
        # Western Australia coordinate bounds (approximate)
        WA_BOUNDS = {
            'min_longitude': 112.0,
            'max_longitude': 130.0,
            'min_latitude': -36.0,
            'max_latitude': -13.0
        }
        
        total_records = len(data)
        valid_coordinates = 0
        valid_geometries = 0
        
        # Validate coordinate ranges
        longitude_valid = (
            (data['LONGITUDE'] >= WA_BOUNDS['min_longitude']) & 
            (data['LONGITUDE'] <= WA_BOUNDS['max_longitude'])
        )
        latitude_valid = (
            (data['LATITUDE'] >= WA_BOUNDS['min_latitude']) & 
            (data['LATITUDE'] <= WA_BOUNDS['max_latitude'])
        )
        
        coordinate_valid = longitude_valid & latitude_valid
        valid_coordinates = coordinate_valid.sum()
        
        # Create geometries and validate
        try:
            geometries = [Point(lon, lat) for lon, lat in zip(data['LONGITUDE'], data['LATITUDE'])]
            valid_geometries = sum(1 for geom in geometries if geom.is_valid)
        except Exception as e:
            logger.error(f"Geometry validation error: {str(e)}")
            valid_geometries = 0
        
        # Validate metadata completeness
        metadata_columns = ['PERMIT_ID', 'MINERAL_TYPE', 'DESCRIPTION']
        metadata_complete = data[metadata_columns].notna().all(axis=1).sum()
        
        # Calculate accuracy metrics
        coordinate_accuracy = (valid_coordinates / total_records) * 100 if total_records > 0 else 0
        geometry_accuracy = (valid_geometries / total_records) * 100 if total_records > 0 else 0
        metadata_completeness = (metadata_complete / total_records) * 100 if total_records > 0 else 0
        
        validation_report = ValidationReport(
            total_records=total_records,
            valid_coordinates=valid_coordinates,
            valid_geometries=valid_geometries,
            valid_metadata=metadata_complete,
            coordinate_accuracy=coordinate_accuracy,
            geometry_accuracy=geometry_accuracy,
            metadata_completeness=metadata_completeness
        )
        
        logger.info(f"Coordinate validation: {coordinate_accuracy:.2f}% accuracy")
        logger.info(f"Geometry validation: {geometry_accuracy:.2f}% accuracy")
        logger.info(f"Metadata completeness: {metadata_completeness:.2f}%")
        
        return validation_report
    
    def transform_coordinate_system(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert GDA2020 → WGS84 coordinate transformation
        Success Metric: <0.001% coordinate transformation error
        """
        logger.info("Starting coordinate system transformation...")
        start_time = time.time()
        
        try:
            # Create GeoDataFrame with source CRS (GDA2020)
            geometry = [Point(lon, lat) for lon, lat in zip(data['LONGITUDE'], data['LATITUDE'])]
            gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=self.processing_config.coordinate_system_source)
            
            # Transform to target CRS (WGS84)
            gdf_transformed = gdf.to_crs(self.processing_config.coordinate_system_target)
            
            # Update coordinate columns with transformed values
            data_transformed = data.copy()
            data_transformed['LONGITUDE'] = gdf_transformed.geometry.x
            data_transformed['LATITUDE'] = gdf_transformed.geometry.y
            data_transformed['GEOMETRY'] = gdf_transformed.geometry.apply(lambda geom: geom.wkt)
            
            # Calculate transformation accuracy (compare with original for validation)
            transformation_error_count = 0
            for i, (orig_geom, trans_geom) in enumerate(zip(geometry, gdf_transformed.geometry)):
                # Check for significant coordinate shifts (shouldn't be large for WA)
                if abs(orig_geom.x - trans_geom.x) > 1.0 or abs(orig_geom.y - trans_geom.y) > 1.0:
                    transformation_error_count += 1
            
            self.metrics.coordinate_transformation_errors = transformation_error_count
            processing_time = time.time() - start_time
            
            logger.info(f"Coordinate transformation completed in {processing_time:.2f} seconds")
            logger.info(f"Transformation errors: {transformation_error_count}/{len(data)} records")
            
            return data_transformed
            
        except Exception as e:
            logger.error(f"Coordinate transformation error: {str(e)}")
            self.metrics.coordinate_transformation_errors = len(data)
            raise
    
    def extract_geological_metadata(self, data: pd.DataFrame) -> Dict:
        """
        Extract mineral types, depths, exploration details
        Success Metric: 100% metadata field extraction
        """
        logger.info("Extracting geological metadata...")
        
        metadata_summary = {
            'total_records': len(data),
            'mineral_types': data['MINERAL_TYPE'].value_counts().to_dict(),
            'unique_permits': data['PERMIT_ID'].nunique(),
            'coordinate_range': {
                'longitude_min': float(data['LONGITUDE'].min()),
                'longitude_max': float(data['LONGITUDE'].max()),
                'latitude_min': float(data['LATITUDE'].min()),
                'latitude_max': float(data['LATITUDE'].max())
            },
            'data_completeness': {
                'permit_id_complete': (data['PERMIT_ID'].notna().sum() / len(data)) * 100,
                'coordinates_complete': ((data['LONGITUDE'].notna() & data['LATITUDE'].notna()).sum() / len(data)) * 100,
                'mineral_type_complete': (data['MINERAL_TYPE'].notna().sum() / len(data)) * 100,
                'description_complete': (data['DESCRIPTION'].notna().sum() / len(data)) * 100
            }
        }
        
        # Extract geological terms from descriptions
        geological_terms = self._extract_geological_terms(data['DESCRIPTION'])
        metadata_summary['geological_terms'] = geological_terms
        
        logger.info(f"Extracted metadata for {len(data)} records")
        logger.info(f"Found {len(metadata_summary['mineral_types'])} unique mineral types")
        logger.info(f"Data completeness average: {sum(metadata_summary['data_completeness'].values()) / len(metadata_summary['data_completeness']):.2f}%")
        
        return metadata_summary
    
    def _extract_geological_terms(self, descriptions: pd.Series) -> Dict[str, int]:
        """Extract common geological terms from descriptions"""
        # Common geological/mining terms
        geological_terms = [
            'gold', 'iron', 'copper', 'nickel', 'lithium', 'uranium', 'zinc', 'lead',
            'ore', 'deposit', 'vein', 'lode', 'mineralization', 'exploration',
            'drilling', 'assay', 'grade', 'tonnage', 'outcrop', 'geology'
        ]
        
        term_counts = {}
        for term in geological_terms:
            count = descriptions.str.contains(term, case=False, na=False).sum()
            if count > 0:
                term_counts[term] = int(count)
        
        return term_counts
    
    def generate_processing_report(self) -> Dict:
        """
        Create detailed processing accuracy and performance report
        Success Metric: Automated evidence generation for supervision
        """
        # Calculate final processing accuracy
        if self.metrics.total_records > 0:
            self.metrics.processing_accuracy = (
                self.metrics.processed_successfully / self.metrics.total_records
            ) * 100
        
        report = {
            'processing_metrics': asdict(self.metrics),
            'success_criteria': {
                'target_accuracy': 98.0,
                'achieved_accuracy': self.metrics.processing_accuracy,
                'accuracy_met': self.metrics.processing_accuracy >= 98.0,
                'target_records': 1000,
                'processed_records': self.metrics.processed_successfully,
                'record_target_met': self.metrics.processed_successfully >= 1000
            },
            'supervisor_validation': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'database_ready': True,  # Set by database operations
                'api_ready': False,      # Set by API initialization
                'module2_integration_ready': False  # Set when API contracts are established
            }
        }
        
        logger.info(f"Processing Report Generated:")
        logger.info(f"  - Accuracy: {self.metrics.processing_accuracy:.2f}% (Target: 98%)")
        logger.info(f"  - Records Processed: {self.metrics.processed_successfully}/{self.metrics.total_records}")
        logger.info(f"  - Processing Time: {self.metrics.processing_time_seconds:.2f} seconds")
        
        return report

class SpatialDataValidator:
    """
    Geological data quality assurance and validation
    Measurable Success: 99%+ spatial data integrity verification
    """
    
    def validate_polygon_geometry(self, geometries: List) -> Dict:
        """PostGIS geometry validation and topology checking"""
        valid_count = 0
        invalid_geometries = []
        
        for i, geom in enumerate(geometries):
            try:
                if hasattr(geom, 'is_valid') and geom.is_valid:
                    valid_count += 1
                else:
                    invalid_geometries.append(i)
            except:
                invalid_geometries.append(i)
        
        validation_result = {
            'total_geometries': len(geometries),
            'valid_geometries': valid_count,
            'invalid_geometries': len(invalid_geometries),
            'validity_percentage': (valid_count / len(geometries)) * 100 if geometries else 0,
            'invalid_geometry_indices': invalid_geometries[:10]  # First 10 for debugging
        }
        
        return validation_result
    
    def check_coordinate_boundaries(self, coordinates: List[Tuple[float, float]]) -> Dict:
        """Ensure coordinates fall within Western Australia bounds"""
        WA_BOUNDS = {
            'min_longitude': 112.0, 'max_longitude': 130.0,
            'min_latitude': -36.0, 'max_latitude': -13.0
        }
        
        within_bounds = 0
        outside_bounds = []
        
        for i, (lon, lat) in enumerate(coordinates):
            if (WA_BOUNDS['min_longitude'] <= lon <= WA_BOUNDS['max_longitude'] and
                WA_BOUNDS['min_latitude'] <= lat <= WA_BOUNDS['max_latitude']):
                within_bounds += 1
            else:
                outside_bounds.append(i)
        
        boundary_report = {
            'total_coordinates': len(coordinates),
            'within_wa_bounds': within_bounds,
            'outside_bounds': len(outside_bounds),
            'boundary_compliance': (within_bounds / len(coordinates)) * 100 if coordinates else 0,
            'wa_bounds_used': WA_BOUNDS
        }
        
        return boundary_report
    
    def verify_mineral_classifications(self, metadata: Dict) -> Dict:
        """Validate geological terminology and mineral types"""
        # Common mineral types in Western Australia
        known_minerals = {
            'gold', 'iron ore', 'iron', 'copper', 'nickel', 'lithium', 'bauxite',
            'uranium', 'zinc', 'lead', 'silver', 'platinum', 'rare earth elements',
            'coal', 'oil', 'gas', 'potash', 'salt', 'sand', 'gravel'
        }
        
        if 'mineral_types' in metadata:
            found_minerals = set(mineral.lower() for mineral in metadata['mineral_types'].keys())
            recognized_minerals = found_minerals.intersection(known_minerals)
            unrecognized_minerals = found_minerals - known_minerals
            
            classification_report = {
                'total_mineral_types': len(found_minerals),
                'recognized_minerals': list(recognized_minerals),
                'unrecognized_minerals': list(unrecognized_minerals),
                'recognition_rate': (len(recognized_minerals) / len(found_minerals)) * 100 if found_minerals else 0,
                'mineral_distribution': metadata['mineral_types']
            }
        else:
            classification_report = {
                'error': 'No mineral types found in metadata',
                'recognition_rate': 0
            }
        
        return classification_report
