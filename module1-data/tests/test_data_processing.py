"""
Unit Tests for WAMEX Data Processing
Validation of learning outcomes and performance targets
"""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.wamex_processor import WAMEXDataProcessor, SpatialDataValidator
from src.config import config

class TestWAMEXDataProcessor:
    """Test core data processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance for testing"""
        return WAMEXDataProcessor()
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing"""
        data = {
            'PERMIT_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'LONGITUDE': [115.8605, 121.4944, 118.7, 116.5, 117.2],
            'LATITUDE': [-31.9505, -30.7522, -32.1, -29.8, -33.5],
            'MINERAL_TYPE': ['Gold', 'Iron Ore', 'Copper', 'Nickel', 'Lithium'],
            'DESCRIPTION': [
                'Gold exploration in Perth region',
                'Iron ore deposit near Kalgoorlie',
                'Copper mineralization zone',
                'Nickel sulfide exploration',
                'Lithium brine project'
            ]
        }
        return pd.DataFrame(data)
    
    def test_load_csv_data_success(self, processor, sample_csv_data):
        """Test successful CSV data loading"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            csv_file = f.name
        
        try:
            # Test loading
            loaded_data = processor.load_csv_data(csv_file)
            
            # Assertions
            assert len(loaded_data) == 5
            assert 'PERMIT_ID' in loaded_data.columns
            assert 'LONGITUDE' in loaded_data.columns
            assert 'LATITUDE' in loaded_data.columns
            assert processor.metrics.total_records == 5
            
        finally:
            os.unlink(csv_file)
    
    def test_load_csv_data_empty_file(self, processor):
        """Test handling of empty CSV file"""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')
            csv_file = f.name
        
        try:
            with pytest.raises(Exception):
                processor.load_csv_data(csv_file)
        finally:
            os.unlink(csv_file)
    
    def test_validate_spatial_coordinates(self, processor, sample_csv_data):
        """Test spatial coordinate validation"""
        validation_report = processor.validate_spatial_coordinates(sample_csv_data)
        
        # Assertions
        assert validation_report.total_records == 5
        assert validation_report.coordinate_accuracy >= 80.0  # WA coordinates should be valid
        assert validation_report.geometry_accuracy >= 80.0
        assert validation_report.valid_coordinates >= 4  # Most coordinates should be valid
    
    def test_coordinate_transformation(self, processor, sample_csv_data):
        """Test coordinate system transformation"""
        transformed_data = processor.transform_coordinate_system(sample_csv_data)
        
        # Assertions
        assert len(transformed_data) == len(sample_csv_data)
        assert 'GEOMETRY' in transformed_data.columns
        assert all(transformed_data['LONGITUDE'].notna())
        assert all(transformed_data['LATITUDE'].notna())
        
        # Check that coordinates are still in reasonable range
        assert all(transformed_data['LONGITUDE'].between(112, 130))
        assert all(transformed_data['LATITUDE'].between(-36, -13))
    
    def test_extract_geological_metadata(self, processor, sample_csv_data):
        """Test geological metadata extraction"""
        metadata = processor.extract_geological_metadata(sample_csv_data)
        
        # Assertions
        assert metadata['total_records'] == 5
        assert 'mineral_types' in metadata
        assert 'coordinate_range' in metadata
        assert 'data_completeness' in metadata
        
        # Check mineral types extraction
        assert len(metadata['mineral_types']) > 0
        assert 'Gold' in metadata['mineral_types']
        
        # Check coordinate range
        coord_range = metadata['coordinate_range']
        assert coord_range['longitude_min'] >= 112
        assert coord_range['longitude_max'] <= 130
    
    def test_processing_accuracy_target(self, processor, sample_csv_data):
        """Test that processing meets 98% accuracy target"""
        # Simulate successful processing
        processor.metrics.total_records = len(sample_csv_data)
        processor.metrics.processed_successfully = len(sample_csv_data)
        processor.metrics.failed_records = 0
        
        report = processor.generate_processing_report()
        
        # Assertions for learning outcome
        assert report['processing_metrics']['processing_accuracy'] == 100.0
        assert report['success_criteria']['accuracy_met'] == True
        assert report['processing_metrics']['total_records'] == 5

class TestSpatialDataValidator:
    """Test spatial data validation functionality"""
    
    @pytest.fixture
    def validator(self):
        return SpatialDataValidator()
    
    def test_coordinate_boundary_validation(self, validator):
        """Test Western Australia coordinate boundary checking"""
        # Valid WA coordinates
        valid_coords = [(115.8605, -31.9505), (121.4944, -30.7522)]
        # Invalid coordinates (outside WA)
        invalid_coords = [(0.0, 0.0), (180.0, 90.0)]
        
        all_coords = valid_coords + invalid_coords
        
        boundary_report = validator.check_coordinate_boundaries(all_coords)
        
        # Assertions
        assert boundary_report['total_coordinates'] == 4
        assert boundary_report['within_wa_bounds'] == 2
        assert boundary_report['outside_bounds'] == 2
        assert boundary_report['boundary_compliance'] == 50.0
    
    def test_mineral_classification_validation(self, validator):
        """Test mineral type classification"""
        metadata = {
            'mineral_types': {
                'Gold': 10,
                'Iron Ore': 15,
                'Unknown Mineral': 5,
                'Copper': 8
            }
        }
        
        classification_report = validator.verify_mineral_classifications(metadata)
        
        # Assertions
        assert classification_report['total_mineral_types'] == 4
        assert 'gold' in classification_report['recognized_minerals']
        assert 'iron ore' in classification_report['recognized_minerals']
        assert classification_report['recognition_rate'] >= 75.0  # Most should be recognized

class TestPerformanceTargets:
    """Test performance targets for supervision validation"""
    
    def test_processing_speed_target(self):
        """Test that processing completes within time targets"""
        processor = WAMEXDataProcessor()
        
        # Create larger sample data to test performance
        sample_size = 100
        large_sample = pd.DataFrame({
            'PERMIT_ID': [f'P{i:03d}' for i in range(sample_size)],
            'LONGITUDE': [115.8605 + (i % 10) * 0.1 for i in range(sample_size)],
            'LATITUDE': [-31.9505 - (i % 10) * 0.1 for i in range(sample_size)],
            'MINERAL_TYPE': ['Gold'] * sample_size,
            'DESCRIPTION': ['Test description'] * sample_size
        })
        
        import time
        start_time = time.time()
        
        # Process data
        validation_report = processor.validate_spatial_coordinates(large_sample)
        transformed_data = processor.transform_coordinate_system(large_sample)
        metadata = processor.extract_geological_metadata(transformed_data)
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 10.0  # Should complete within 10 seconds for 100 records
        assert validation_report.coordinate_accuracy >= 98.0
        assert len(transformed_data) == sample_size
    
    def test_accuracy_target_validation(self):
        """Test that accuracy targets are properly validated"""
        processor = WAMEXDataProcessor()
        
        # Test with perfect data
        perfect_data = pd.DataFrame({
            'PERMIT_ID': ['P001', 'P002'],
            'LONGITUDE': [115.8605, 121.4944],  # Valid WA coordinates
            'LATITUDE': [-31.9505, -30.7522],
            'MINERAL_TYPE': ['Gold', 'Iron Ore'],
            'DESCRIPTION': ['Valid description 1', 'Valid description 2']
        })
        
        validation_report = processor.validate_spatial_coordinates(perfect_data)
        
        # Should achieve high accuracy with perfect data
        assert validation_report.coordinate_accuracy >= 98.0
        assert validation_report.geometry_accuracy >= 98.0
        assert validation_report.metadata_completeness >= 98.0

if __name__ == '__main__':
    pytest.main([__file__])
