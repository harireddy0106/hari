# src/utils/helpers.py
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import netCDF4 as nc
import hashlib
import re

class ArgoHelpers:
    """Helper functions for Argo data processing"""
    
    @staticmethod
    def parse_juld(juld: float, ref_date: datetime = datetime(1950, 1, 1)) -> datetime:
        """Convert Julian date to datetime object"""
        try:
            return ref_date + timedelta(days=float(juld))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        try:
            R = 6371  # Earth radius in km
            
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            
            a = (np.sin(dlat/2) * np.sin(dlat/2) + 
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
                 np.sin(dlon/2) * np.sin(dlon/2))
            
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        except (ValueError, TypeError):
            return float('inf')
    
    @staticmethod
    def normalize_pressure(pressure: List[float]) -> List[float]:
        """Normalize pressure values to remove outliers"""
        if not pressure:
            return []
        
        pressure_array = np.array(pressure)
        pressure_array = pressure_array[pressure_array >= 0]  # Remove negative pressures
        pressure_array = pressure_array[pressure_array <= 4000]  # Remove unrealistic high pressures
        
        return pressure_array.tolist()
    
    @staticmethod
    def interpolate_to_standard_levels(values: List[float], 
                                     original_pressures: List[float],
                                     standard_pressures: List[float] = None) -> List[float]:
        """Interpolate values to standard pressure levels"""
        if not standard_pressures:
            standard_pressures = list(range(0, 2001, 10))  # 0-2000 dbar in 10 dbar steps
        
        if not values or not original_pressures:
            return [None] * len(standard_pressures)
        
        # Remove None values
        valid_mask = [v is not None for v in values]
        valid_pressures = np.array(original_pressures)[valid_mask]
        valid_values = np.array(values)[valid_mask]
        
        if len(valid_values) < 2:
            return [None] * len(standard_pressures)
        
        # Interpolate
        try:
            interpolated = np.interp(standard_pressures, valid_pressures, valid_values)
            return interpolated.tolist()
        except ValueError:
            return [None] * len(standard_pressures)
    
    @staticmethod
    def calculate_mixed_layer_depth(temperature: List[float], 
                                  pressure: List[float], 
                                  threshold: float = 0.2) -> float:
        """Calculate mixed layer depth from temperature profile"""
        if not temperature or not pressure:
            return None
        
        # Find surface temperature
        surface_temp = next((t for t in temperature if t is not None), None)
        if surface_temp is None:
            return None
        
        # Find depth where temperature changes by threshold
        for i, (temp, press) in enumerate(zip(temperature, pressure)):
            if temp is not None and abs(temp - surface_temp) >= threshold:
                return press
        
        return pressure[-1] if pressure else None

class DataValidator:
    """Data validation and quality control"""
    
    def __init__(self):
        self.quality_rules = self._initialize_quality_rules()
    
    def _initialize_quality_rules(self) -> Dict[str, Any]:
        """Initialize data quality validation rules"""
        return {
            'temperature': {'min': -2, 'max': 40},
            'salinity': {'min': 0, 'max': 42},
            'pressure': {'min': 0, 'max': 4000},
            'oxygen': {'min': 0, 'max': 500},
            'chlorophyll': {'min': 0, 'max': 50}
        }
    
    def validate_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete profile"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0
        }
        
        # Check required fields
        required_fields = ['latitude', 'longitude', 'profile_date']
        for field in required_fields:
            if not profile_data.get(field):
                validation_results['errors'].append(f"Missing required field: {field}")
                validation_results['is_valid'] = False
        
        # Validate coordinate ranges
        if not (-90 <= profile_data.get('latitude', 100) <= 90):
            validation_results['errors'].append("Latitude out of range (-90 to 90)")
            validation_results['is_valid'] = False
        
        if not (-180 <= profile_data.get('longitude', 200) <= 180):
            validation_results['errors'].append("Longitude out of range (-180 to 180)")
            validation_results['is_valid'] = False
        
        # Validate variable ranges
        for var_name, rules in self.quality_rules.items():
            values = profile_data.get(f'{var_name}_values', [])
            valid_values = [v for v in values if v is not None]
            
            for value in valid_values:
                if not (rules['min'] <= value <= rules['max']):
                    validation_results['warnings'].append(
                        f"{var_name} value {value} outside expected range"
                    )
        
        # Calculate quality score
        total_checks = len(validation_results['errors']) + len(validation_results['warnings']) + 1
        error_penalty = len(validation_results['errors']) * 0.5
        warning_penalty = len(validation_results['warnings']) * 0.1
        
        validation_results['quality_score'] = max(0, 1.0 - error_penalty - warning_penalty)
        
        return validation_results
    
    def detect_spikes(self, values: List[float], threshold: float = 3.0) -> List[bool]:
        """Detect spikes in data using median absolute deviation"""
        if not values or len(values) < 3:
            return [False] * len(values) if values else []
        
        valid_values = [v for v in values if v is not None]
        if len(valid_values) < 3:
            return [False] * len(values)
        
        median = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median))
        
        if mad == 0:
            return [False] * len(values)
        
        spikes = []
        for value in values:
            if value is None:
                spikes.append(False)
            else:
                z_score = 0.6745 * (value - median) / mad  # Convert to z-score
                spikes.append(abs(z_score) > threshold)
        
        return spikes

class DataTransformer:
    """Data transformation utilities"""
    
    @staticmethod
    def profile_to_dataframe(profile_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert profile data to pandas DataFrame"""
        records = []
        
        pressure_levels = profile_data.get('pressure_levels', [])
        n_levels = len(pressure_levels)
        
        for i in range(n_levels):
            record = {
                'pressure': pressure_levels[i] if i < len(pressure_levels) else None,
                'temperature': profile_data.get('temperature_values', [])[i] if i < len(profile_data.get('temperature_values', [])) else None,
                'salinity': profile_data.get('salinity_values', [])[i] if i < len(profile_data.get('salinity_values', [])) else None,
                'oxygen': profile_data.get('oxygen_values', [])[i] if i < len(profile_data.get('oxygen_values', [])) else None,
                'chlorophyll': profile_data.get('chlorophyll_values', [])[i] if i < len(profile_data.get('chlorophyll_values', [])) else None,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    @staticmethod
    def calculate_derived_variables(temperature: List[float], salinity: List[float], 
                                  pressure: List[float]) -> Dict[str, List[float]]:
        """Calculate derived oceanographic variables"""
        # Placeholder for derived variables calculation
        # This would implement calculations for density, sound speed, etc.
        return {
            'density': [None] * len(temperature),
            'sound_speed': [None] * len(temperature)
        }

class FileHandler:
    """File handling utilities"""
    
    @staticmethod
    def generate_file_hash(file_path: Path) -> str:
        """Generate MD5 hash for file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logging.error(f"Error generating file hash: {e}")
            return ""
    
    @staticmethod
    def safe_json_serialize(data: Any) -> str:
        """Safely serialize data to JSON handling numpy types"""
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(data, default=default_serializer)
    
    @staticmethod
    def parse_filename_pattern(filename: str) -> Dict[str, str]:
        """Parse Argo filename pattern to extract metadata"""
        patterns = [
            r'(?P<dac>\w+)/(?P<float_id>\d+)/profiles/(?P<mode>[RD])?(?P<file_type>\w+)\.nc',
            r'(?P<float_id>\d+)_(?P<cycle>\d+)\.nc'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                return match.groupdict()
        
        return {}