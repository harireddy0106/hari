# tests/test_database.py

import pytest
import tempfile
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from database.database_manager import db_manager
from database.models import ArgoFloat, ArgoProfile, UserQuery, SystemLog
from config import config

class TestDatabaseManager:
    """Test cases for database manager"""
    
    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Setup temporary database for testing"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_url = f"sqlite:///{self.temp_db.name}"
        
        # Initialize database
        config.set('database.url', self.db_url)
        config.set('database.echo', False)
        
        db_manager.initialize_database(self.db_url, echo=False)
        db_manager.create_tables()
        
        yield
        
        # Cleanup
        db_manager.close_connections()
        self.temp_db.close()
        Path(self.temp_db.name).unlink()
    
    def test_database_initialization(self):
        """Test database initialization"""
        assert db_manager.engine is not None
        assert db_manager.SessionLocal is not None
        
        # Test connection
        with db_manager.get_session() as session:
            result = session.execute("SELECT 1").scalar()
            assert result == 1
    
    def test_table_creation(self):
        """Test that tables are created correctly"""
        with db_manager.get_session() as session:
            # Check if tables exist
            tables = session.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'argo_%'
            """).fetchall()
            
            table_names = [t[0] for t in tables]
            expected_tables = ['argo_floats', 'argo_profiles']
            
            for table in expected_tables:
                assert table in table_names
    
    def test_argo_float_operations(self):
        """Test ArgoFloat model operations"""
        with db_manager.get_session() as session:
            # Create test float
            test_float = ArgoFloat(
                float_id="test_float_001",
                platform_number="1900001",
                data_center="TEST",
                institution="Test Institution",
                date_created=datetime.utcnow(),
                is_active=True,
                last_latitude=15.5,
                last_longitude=65.3,
                sensor_types=json.dumps(["TEMP", "PSAL", "PRES"]),
                max_depth=2000.0,
                data_mode="R",
                quality_score=95.0
            )
            
            session.add(test_float)
            session.commit()
            
            # Retrieve and verify
            retrieved_float = session.query(ArgoFloat).filter_by(float_id="test_float_001").first()
            assert retrieved_float is not None
            assert retrieved_float.platform_number == "1900001"
            assert retrieved_float.is_active == True
            assert retrieved_float.quality_score == 95.0
    
    def test_argo_profile_operations(self):
        """Test ArgoProfile model operations"""
        with db_manager.get_session() as session:
            # First create a float
            test_float = ArgoFloat(
                float_id="test_float_001",
                platform_number="1900001",
                data_center="TEST",
                is_active=True
            )
            session.add(test_float)
            session.commit()
            
            # Create test profile
            test_profile = ArgoProfile(
                float_id="test_float_001",
                cycle_number=1,
                profile_date=datetime.utcnow(),
                latitude=15.5,
                longitude=65.3,
                pressure_levels=json.dumps([0, 10, 20, 30]),
                temperature_values=json.dumps([25.0, 24.5, 23.0, 22.0]),
                salinity_values=json.dumps([35.0, 35.1, 35.2, 35.3]),
                data_mode="R",
                quality_score=88.5,
                source_file="test_file.nc",
                file_hash="test_hash"
            )
            
            session.add(test_profile)
            session.commit()
            
            # Retrieve and verify
            retrieved_profile = session.query(ArgoProfile).filter_by(
                float_id="test_float_001", 
                cycle_number=1
            ).first()
            
            assert retrieved_profile is not None
            assert retrieved_profile.latitude == 15.5
            assert retrieved_profile.quality_score == 88.5
            assert len(json.loads(retrieved_profile.temperature_values)) == 4
    
    def test_relationship_integrity(self):
        """Test relationship integrity between floats and profiles"""
        with db_manager.get_session() as session:
            # Create float and profile
            test_float = ArgoFloat(
                float_id="test_float_002",
                platform_number="1900002",
                is_active=True
            )
            session.add(test_float)
            session.commit()
            
            test_profile = ArgoProfile(
                float_id="test_float_002",
                cycle_number=1,
                profile_date=datetime.utcnow(),
                latitude=20.0,
                longitude=70.0
            )
            session.add(test_profile)
            session.commit()
            
            # Verify relationship
            float_with_profiles = session.query(ArgoFloat).filter_by(float_id="test_float_002").first()
            assert float_with_profiles is not None
            # Note: This would need proper relationship setup in models
    
    def test_database_stats(self):
        """Test database statistics function"""
        # Add some test data
        with db_manager.get_session() as session:
            for i in range(3):
                float_obj = ArgoFloat(
                    float_id=f"stats_test_{i}",
                    platform_number=f"1900{i}",
                    is_active=True
                )
                session.add(float_obj)
                
                profile_obj = ArgoProfile(
                    float_id=f"stats_test_{i}",
                    cycle_number=1,
                    profile_date=datetime.utcnow(),
                    latitude=10.0 + i,
                    longitude=60.0 + i
                )
                session.add(profile_obj)
            
            session.commit()
        
        # Get statistics
        stats = db_manager.get_database_stats()
        
        assert 'argo_floats_count' in stats
        assert 'argo_profiles_count' in stats
        assert stats['argo_floats_count'] >= 3
        assert stats['argo_profiles_count'] >= 3
    
    def test_error_handling(self):
        """Test database error handling"""
        # Test invalid database URL
        with pytest.raises(Exception):
            db_manager.initialize_database("invalid://url")
        
        # Test duplicate float ID
        with db_manager.get_session() as session:
            float1 = ArgoFloat(float_id="duplicate_test", platform_number="1900001", is_active=True)
            session.add(float1)
            session.commit()
            
            # Try to add duplicate
            float2 = ArgoFloat(float_id="duplicate_test", platform_number="1900002", is_active=True)
            session.add(float2)
            
            # This should raise an integrity error
            with pytest.raises(Exception):
                session.commit()

class TestDataValidation:
    """Test data validation scenarios"""
    
    def test_coordinate_validation(self):
        """Test coordinate validation logic"""
        from utils.helpers import ArgoHelpers
        
        # Valid coordinates
        assert ArgoHelpers.validate_coordinates(15.5, 65.3) == True
        assert ArgoHelpers.validate_coordinates(-45.0, -120.0) == True
        assert ArgoHelpers.validate_coordinates(90.0, 180.0) == True
        
        # Invalid coordinates
        assert ArgoHelpers.validate_coordinates(91.0, 65.3) == False  # Latitude too high
        assert ArgoHelpers.validate_coordinates(-91.0, 65.3) == False  # Latitude too low
        assert ArgoHelpers.validate_coordinates(45.0, 181.0) == False  # Longitude too high
        assert ArgoHelpers.validate_coordinates(45.0, -181.0) == False  # Longitude too low
    
    def test_date_parsing(self):
        """Test date parsing functionality"""
        from utils.helpers import ArgoHelpers
        
        # Valid date strings
        test_date = datetime(2023, 1, 15, 10, 30, 0)
        date_str = "2023-01-15T10:30:00"
        parsed = ArgoHelpers.parse_date_string(date_str)
        assert parsed == test_date
        
        # Invalid date strings
        assert ArgoHelpers.parse_date_string("invalid_date") is None
        assert ArgoHelpers.parse_date_string("") is None
        assert ArgoHelpers.parse_date_string(None) is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])