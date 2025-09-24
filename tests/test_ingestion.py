# tests/test_ingestion.py

import pytest
import tempfile
from pathlib import Path
import sys
import xarray as xr
import numpy as np
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.argo_data_ingestor import ArgoDataIngestor
from data.quality_control import ArgoQualityControl
from database.database_manager import db_manager
from config import config

class TestDataIngestion:
    """Test cases for data ingestion functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_url = f"sqlite:///{self.temp_db.name}"
        
        # Initialize database
        config.set('database.url', self.db_url)
        config.set('database.echo', False)
        
        db_manager.initialize_database(self.db_url, echo=False)
        db_manager.create_tables()
        
        # Initialize ingestor
        self.ingestor = ArgoDataIngestor()
        
        yield
        
        # Cleanup
        db_manager.close_connections()
        self.temp_db.close()
        Path(self.temp_db.name).unlink()
    
    def create_sample_netcdf(self, file_path, n_profiles=3, n_levels=50):
        """Create a sample NetCDF file for testing"""
        # Create sample data
        latitudes = np.random.uniform(10, 25, n_profiles)
        longitudes = np.random.uniform(60, 85, n_profiles)
        times = np.arange(n_profiles) * 10  # Days since reference
        
        # Create pressure and temperature data
        pressure = np.tile(np.linspace(0, 500, n_levels), (n_profiles, 1))
        temperature = 10 + 10 * np.exp(-pressure / 100) + np.random.normal(0, 0.1, pressure.shape)
        
        # Create xarray dataset
        ds = xr.Dataset({
            'LATITUDE': (['N_PROF'], latitudes),
            'LONGITUDE': (['N_PROF'], longitudes),
            'JULD': (['N_PROF'], times),
            'PRES': (['N_PROF', 'N_LEVELS'], pressure),
            'TEMP': (['N_PROF', 'N_LEVELS'], temperature),
            'PSAL': (['N_PROF', 'N_LEVELS'], 35.0 + 0.1 * np.sin(pressure/100)),
        })
        
        # Add required global attributes
        ds.attrs = {
            'DATA_TYPE': 'ARGO PROFILE',
            'FORMAT_VERSION': '3.1',
            'DATA_CENTRE': 'CORIOLIS',
            'PLATFORM_NUMBER': '1901234',
            'DATE_CREATION': '20230115000000'
        }
        
        # Save to file
        ds.to_netcdf(file_path)
        return file_path
    
    def test_netcdf_processing(self):
        """Test NetCDF file processing"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            sample_file = Path(temp_file.name)
            self.create_sample_netcdf(sample_file)
            
            # Process the file
            result = self.ingestor.process_netcdf_file(sample_file)
            
            # Verify results
            assert result is not None
            assert 'floats' in result
            assert 'profiles' in result
            assert len(result['profiles']) == 3  # Should have 3 profiles
            
            # Verify float data
            float_data = result['floats'][0]
            assert float_data['float_id'] == '1901234'
            assert float_data['platform_number'] == '1901234'
            
            # Verify profile data
            profile = result['profiles'][0]
            assert 'latitude' in profile
            assert 'longitude' in profile
            assert 'temperature_values' in profile
            assert 'pressure_levels' in profile
            
            # Cleanup
            sample_file.unlink()
    
    def test_quality_control_applied(self):
        """Test that quality control is applied during processing"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            sample_file = Path(temp_file.name)
            self.create_sample_netcdf(sample_file)
            
            # Process with quality control
            result = self.ingestor.process_netcdf_file(sample_file)
            
            # Verify QC is applied
            for profile in result['profiles']:
                assert 'quality_score' in profile
                assert 'qc_results' in profile
                assert 'data_quality' in profile
                
                # Quality score should be between 0 and 100
                assert 0 <= profile['quality_score'] <= 100
            
            sample_file.unlink()
    
    def test_file_hash_calculation(self):
        """Test file hash calculation for duplicate detection"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            sample_file = Path(temp_file.name)
            self.create_sample_netcdf(sample_file)
            
            # Calculate hash
            hash1 = self.ingestor._calculate_file_hash(sample_file)
            hash2 = self.ingestor._calculate_file_hash(sample_file)
            
            # Hashes should be identical for same file
            assert hash1 == hash2
            assert len(hash1) == 32  # MD5 hash length
            
            sample_file.unlink()
    
    def test_duplicate_file_detection(self):
        """Test duplicate file detection"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            sample_file = Path(temp_file.name)
            self.create_sample_netcdf(sample_file)
            
            # Process file first time
            hash1 = self.ingestor._calculate_file_hash(sample_file)
            self.ingestor.processed_files.add(hash1)
            
            # Check if file is detected as processed
            assert self.ingestor._is_file_processed(hash1) == True
            
            # Different file should not be detected as processed
            assert self.ingestor._is_file_processed("different_hash") == False
            
            sample_file.unlink()
    
    def test_data_storage(self):
        """Test data storage in database"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            sample_file = Path(temp_file.name)
            self.create_sample_netcdf(sample_file)
            
            # Process and store data
            result = self.ingestor.process_netcdf_file(sample_file)
            file_hash = self.ingestor._calculate_file_hash(sample_file)
            
            self.ingestor.store_netcdf_data(result, str(sample_file), file_hash)
            
            # Verify data was stored
            with db_manager.get_session() as session:
                from database.models import ArgoFloat, ArgoProfile
                
                # Check float was stored
                stored_float = session.query(ArgoFloat).filter_by(float_id='1901234').first()
                assert stored_float is not None
                assert stored_float.platform_number == '1901234'
                
                # Check profiles were stored
                stored_profiles = session.query(ArgoProfile).filter_by(float_id='1901234').all()
                assert len(stored_profiles) == 3
                
                # Verify profile data integrity
                profile = stored_profiles[0]
                assert profile.latitude is not None
                assert profile.longitude is not None
                assert profile.quality_score is not None
            
            sample_file.unlink()
    
    def test_invalid_file_handling(self):
        """Test handling of invalid files"""
        # Test with non-existent file
        result = self.ingestor.process_netcdf_file(Path("nonexistent.nc"))
        assert result is None
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            empty_file = Path(temp_file.name)
            # Create empty file
            empty_file.write_bytes(b'')
            
            result = self.ingestor.process_netcdf_file(empty_file)
            assert result is None
            
            empty_file.unlink()
    
    def test_metadata_extraction(self):
        """Test NetCDF metadata extraction"""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            sample_file = Path(temp_file.name)
            self.create_sample_netcdf(sample_file)
            
            with xr.open_dataset(sample_file) as ds:
                metadata = self.ingestor._extract_netcdf_metadata(ds, sample_file)
                
                # Verify metadata structure
                assert 'file_name' in metadata
                assert 'file_size' in metadata
                assert 'dataset_dimensions' in metadata
                assert 'global_attributes' in metadata
                assert 'quality_indicators' in metadata
                
                # Verify specific metadata values
                assert metadata['file_name'] == sample_file.name
                assert 'N_PROF' in metadata['dataset_dimensions']
                assert metadata['quality_indicators']['has_temperature'] == True
            
            sample_file.unlink()

class TestQualityControl:
    """Test cases for quality control functionality"""
    
    def setup_method(self):
        """Setup test method"""
        self.qc = ArgoQualityControl()
    
    def test_temperature_range_check(self):
        """Test temperature range validation"""
        # Valid temperatures
        valid_temps = [0, 10, 20, 30, 40]
        result = self.qc._check_temperature_range(valid_temps)
        assert result['passed'] == True
        
        # Invalid temperatures
        invalid_temps = [-5, 10, 20, 50]  # -5 and 50 are outside typical range
        result = self.qc._check_temperature_range(invalid_temps)
        assert result['passed'] == False
        assert 'out_of_range' in result['details']
    
    def test_salinity_range_check(self):
        """Test salinity range validation"""
        # Valid salinity
        valid_salinity = [30, 35, 40]
        result = self.qc._check_salinity_range(valid_salinity)
        assert result['passed'] == True
        
        # Invalid salinity
        invalid_salinity = [0, 35, 50]  # 0 and 50 are outside typical range
        result = self.qc._check_salinity_range(invalid_salinity)
        assert result['passed'] == False
    
    def test_pressure_monotonic_check(self):
        """Test pressure monotonicity check"""
        # Valid pressure (increasing)
        valid_pressure = [0, 10, 20, 30, 40]
        result = self.qc._check_pressure_monotonic(valid_pressure)
        assert result['passed'] == True
        
        # Invalid pressure (non-monotonic)
        invalid_pressure = [0, 10, 5, 20, 30]  # 5 is less than 10
        result = self.qc._check_pressure_monotonic(invalid_pressure)
        assert result['passed'] == False
    
    def test_spike_detection(self):
        """Test spike detection in data"""
        # Smooth data (no spikes)
        smooth_data = [10, 10.1, 10.2, 10.3, 10.4]
        result = self.qc._check_spikes(smooth_data)
        assert result['passed'] == True
        
        # Data with spike
        spike_data = [10, 10.1, 15.0, 10.3, 10.4]  # 15.0 is a spike
        result = self.qc._check_spikes(spike_data)
        assert result['passed'] == False
        assert result['spike_count'] > 0
    
    def test_comprehensive_qc(self):
        """Test comprehensive quality control check"""
        profile_data = {
            'temperature_values': [10, 11, 12, 13, 14],
            'salinity_values': [35, 35.1, 35.2, 35.3, 35.4],
            'pressure_levels': [0, 10, 20, 30, 40],
            'oxygen_values': [200, 210, 220, 230, 240]
        }
        
        result = self.qc.comprehensive_qc_check(profile_data)
        
        # Verify result structure
        assert 'quality_score' in result
        assert 'passed_checks' in result
        assert 'failed_checks' in result
        assert 'details' in result
        
        # Quality score should be calculated
        assert 0 <= result['quality_score'] <= 100
        
        # Should have some passed checks for valid data
        assert len(result['passed_checks']) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])