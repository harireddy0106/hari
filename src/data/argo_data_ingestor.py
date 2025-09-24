
#src/data/argo_data_ingestor.py

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import ftplib
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from database.database_manager import db_manager
from database.models import ArgoFloat, ArgoProfile
from .quality_control import ArgoQualityControl
import json
import os
import hashlib

logger = logging.getLogger(__name__)

class ArgoDataIngestor:
    def __init__(self):
        self.supported_parameters = ['TEMP', 'PSAL', 'PRES', 'DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL']
        self.netcdf_extensions = ['.nc', '.nc4']
        self.qc_engine = ArgoQualityControl()
        self.processed_files = set()
    
    def initialize_processed_files_cache(self):
        """Initialize cache of already processed files"""
        try:
            with db_manager.get_session() as session:
                profiles = session.query(ArgoProfile.source_file).distinct().all()
                self.processed_files = {p[0] for p in profiles if p[0]}
                logger.info(f"Loaded {len(self.processed_files)} processed files from database")
        except Exception as e:
            logger.warning(f"Could not load processed files cache: {e}")
            self.processed_files = set()

    def process_netcdf_directory(self, netcdf_dir: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process all NetCDF files in a directory with enhanced features"""
        self.initialize_processed_files_cache()
        
        netcdf_path = Path(netcdf_dir)
        results = {
            'processed_files': 0,
            'successful_files': 0,
            'errors': [],
            'floats_added': 0,
            'profiles_added': 0,
            'quality_scores': [],
            'start_time': datetime.utcnow(),
            'end_time': None
        }
        
        # Find all NetCDF files recursively
        netcdf_files = []
        for ext in self.netcdf_extensions:
            netcdf_files.extend(netcdf_path.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(netcdf_files)} NetCDF files in {netcdf_dir}")
        
        # Filter out already processed files unless force reprocess
        if not force_reprocess:
            netcdf_files = [f for f in netcdf_files if str(f) not in self.processed_files]
            logger.info(f"After filtering processed files: {len(netcdf_files)} files to process")
        
        # Process files with progress tracking
        for i, nc_file in enumerate(netcdf_files):
            try:
                logger.info(f"Processing file {i+1}/{len(netcdf_files)}: {nc_file.name}")
                
                file_hash = self._calculate_file_hash(nc_file)
                if self._is_file_processed(file_hash) and not force_reprocess:
                    logger.info(f"Skipping already processed file: {nc_file.name}")
                    results['processed_files'] += 1
                    continue
                
                file_results = self.process_netcdf_file(nc_file)
                if file_results:
                    # Apply quality control
                    file_results = self._apply_quality_control(file_results)
                    
                    self.store_netcdf_data(file_results, str(nc_file), file_hash)
                    results['successful_files'] += 1
                    results['floats_added'] += len(file_results.get('floats', []))
                    results['profiles_added'] += len(file_results.get('profiles', []))
                    
                    # Collect quality scores
                    for profile in file_results.get('profiles', []):
                        if 'quality_score' in profile:
                            results['quality_scores'].append(profile['quality_score'])
                
                results['processed_files'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {nc_file}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        results['end_time'] = datetime.utcnow()
        results['processing_duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # Calculate statistics
        if results['quality_scores']:
            results['avg_quality_score'] = np.mean(results['quality_scores'])
            results['min_quality_score'] = np.min(results['quality_scores'])
            results['max_quality_score'] = np.max(results['quality_scores'])
        
        return results

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return str(file_path)  # Fallback to file path

    def _is_file_processed(self, file_hash: str) -> bool:
        """Check if file has already been processed"""
        return file_hash in self.processed_files

    def _apply_quality_control(self, processed_data: Dict) -> Dict:
        """Apply quality control to processed data"""
        for profile in processed_data.get('profiles', []):
            qc_results = self.qc_engine.comprehensive_qc_check(profile)
            profile['qc_results'] = qc_results
            profile['quality_score'] = qc_results['quality_score']
            profile['data_quality'] = self._get_quality_category(qc_results['quality_score'])
        
        return processed_data

    def _get_quality_category(self, score: float) -> str:
        """Convert quality score to category"""
        if score >= 90: return "Excellent"
        elif score >= 80: return "Good"
        elif score >= 70: return "Fair"
        elif score >= 60: return "Poor"
        else: return "Unusable"

    def process_netcdf_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Enhanced NetCDF processing with better error handling"""
        try:
            logger.info(f"Processing NetCDF file: {file_path}")
            
            # Check file size and accessibility
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return None
                
            if file_path.stat().st_size == 0:
                logger.error(f"File is empty: {file_path}")
                return None
            
            # Open NetCDF file with context manager
            with xr.open_dataset(file_path) as ds:
                # Enhanced file type detection
                if self._is_argo_netcdf_file(ds):
                    return self._process_argo_netcdf(ds, file_path)
                else:
                    logger.warning(f"File {file_path} doesn't appear to be standard ARGO data")
                    # Try alternative processing methods
                    return self._try_alternative_processing(ds, file_path)
                    
        except Exception as e:
            logger.error(f"Error reading NetCDF file {file_path}: {e}")
            return None

    def _is_argo_netcdf_file(self, dataset: xr.Dataset) -> bool:
        """Check if dataset is a standard ARGO NetCDF file"""
        argo_indicators = ['PLATFORM_NUMBER', 'JULD', 'LATITUDE', 'LONGITUDE', 'PRES', 'TEMP', 'PSAL']
        return any(hasattr(dataset, attr) or attr in dataset.variables for attr in argo_indicators)

    def _try_alternative_processing(self, dataset: xr.Dataset, file_path: Path) -> Optional[Dict]:
        """Try alternative processing methods for non-standard files"""
        try:
            # Look for common oceanographic data patterns
            ocean_vars = ['temperature', 'salinity', 'pressure', 'depth', 'temp', 'sal', 'pres']
            found_vars = [var for var in dataset.variables if any(ocean in var.lower() for ocean in ocean_vars)]
            
            if len(found_vars) >= 2:  # At least two oceanographic variables
                return self._process_generic_ocean_data(dataset, file_path, found_vars)
            
        except Exception as e:
            logger.warning(f"Alternative processing failed for {file_path}: {e}")
        
        return None

    def _process_generic_ocean_data(self, dataset: xr.Dataset, file_path: Path, ocean_vars: List[str]) -> Dict:
        """Process generic oceanographic data"""
        results = {
            'metadata': self._extract_netcdf_metadata(dataset, file_path),
            'floats': [{'float_id': 'generic', 'platform_number': 'generic'}],
            'profiles': [],
            'processing_notes': ['Processed as generic ocean data']
        }
        
        # Create a generic profile
        profile = {
            'float_id': 'generic',
            'cycle_number': 1,
            'profile_date': datetime.utcnow(),
            'latitude': 0.0,
            'longitude': 0.0,
            'source_file': str(file_path),
            'data_mode': 'R'
        }
        
        # Extract available data
        for var in ocean_vars:
            if hasattr(dataset, var):
                values = getattr(dataset, var).values
                if values is not None:
                    profile[f'{var}_values'] = values.flatten().tolist()[:100]  # Limit size
        
        results['profiles'].append(profile)
        return results

    def store_netcdf_data(self, processed_data: Dict[str, Any], file_path: str, file_hash: str):
        """Enhanced data storage with transaction management"""
        try:
            with db_manager.get_session() as session:
                # Store floats with conflict resolution
                for float_data in processed_data.get('floats', []):
                    argo_float = ArgoFloat(
                        float_id=float_data['float_id'],
                        platform_number=float_data['platform_number'],
                        data_center=float_data.get('data_center', 'Unknown'),
                        institution=float_data.get('institution', 'Unknown'),
                        date_created=float_data.get('date_created', datetime.utcnow()),
                        is_active=True,
                        last_latitude=float_data.get('last_latitude'),
                        last_longitude=float_data.get('last_longitude'),
                        sensor_types=json.dumps(float_data.get('sensor_types', [])),
                        max_depth=float_data.get('max_depth', 2000),
                        data_mode=float_data.get('data_mode', 'R'),
                        quality_score=float_data.get('quality_score', 100.0)
                    )
                    
                    # Use merge to handle duplicates
                    session.merge(argo_float)
                
                # Store profiles
                for profile_data in processed_data.get('profiles', []):
                    profile = ArgoProfile(
                        float_id=profile_data['float_id'],
                        cycle_number=profile_data['cycle_number'],
                        profile_date=profile_data['profile_date'],
                        latitude=profile_data['latitude'],
                        longitude=profile_data['longitude'],
                        pressure_levels=json.dumps(profile_data.get('pressure_levels', [])),
                        temperature_values=json.dumps(profile_data.get('temperature_values', [])),
                        salinity_values=json.dumps(profile_data.get('salinity_values', [])),
                        oxygen_values=json.dumps(profile_data.get('oxygen_values', [])),
                        chlorophyll_values=json.dumps(profile_data.get('chlorophyll_values', [])),
                        nitrate_values=json.dumps(profile_data.get('nitrate_values', [])),
                        data_mode=profile_data.get('data_mode', 'R'),
                        quality_score=profile_data.get('quality_score', 100.0),
                        qc_comments=json.dumps(profile_data.get('qc_results', {})),
                        source_file=file_path,
                        file_hash=file_hash  # Store hash for duplicate detection
                    )
                    session.add(profile)
                
                session.commit()
                logger.info(f"Successfully stored data from {file_path}")
                
                # Add to processed files cache
                self.processed_files.add(file_hash)
                
        except Exception as e:
            logger.error(f"Error storing data from {file_path}: {e}")
            raise

    # Keep existing methods but add enhancements...
    def _extract_netcdf_metadata(self, dataset: xr.Dataset, file_path: Path) -> Dict[str, Any]:
        """Enhanced metadata extraction"""
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_hash': self._calculate_file_hash(file_path),
            'processing_date': datetime.utcnow().isoformat(),
            'dataset_dimensions': dict(dataset.dims),
            'dataset_variables': list(dataset.variables.keys()),
            'global_attributes': {},
            'quality_indicators': {}
        }
        
        # Extract global attributes with better handling
        for attr_name in dataset.attrs:
            try:
                attr_value = str(dataset.attrs[attr_name])
                # Truncate very long attributes
                if len(attr_value) > 1000:
                    attr_value = attr_value[:1000] + "...[truncated]"
                metadata['global_attributes'][attr_name] = attr_value
            except Exception as e:
                metadata['global_attributes'][attr_name] = f"Error reading: {str(e)}"
        
        # Add quality indicators
        metadata['quality_indicators'] = {
            'has_temperature': 'TEMP' in dataset.variables,
            'has_salinity': 'PSAL' in dataset.variables,
            'has_pressure': 'PRES' in dataset.variables,
            'profile_count': len(dataset.N_PROF) if 'N_PROF' in dataset.dims else 1,
            'data_mode': getattr(dataset, 'DATA_MODE', 'Unknown')
        }
        
        return metadata
    
    def _extract_platform_data(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Extract float/platform information from dataset"""
        platform_data = {
            'float_id': self._get_platform_number(dataset),
            'platform_number': self._get_platform_number(dataset),
            'data_center': getattr(dataset, 'DATA_CENTRE', 'Unknown'),
            'institution': getattr(dataset, 'INSTITUTION', 'Unknown'),
            'date_created': self._parse_date(getattr(dataset, 'DATE_CREATION', datetime.utcnow())),
            'last_latitude': float(dataset.LATITUDE.values[0]) if hasattr(dataset, 'LATITUDE') else None,
            'last_longitude': float(dataset.LONGITUDE.values[0]) if hasattr(dataset, 'LONGITUDE') else None,
            'sensor_types': self._extract_sensor_types(dataset),
            'max_depth': self._calculate_max_depth(dataset),
            'data_mode': getattr(dataset, 'DATA_MODE', 'R'),
            'wmo_inst_type': getattr(dataset, 'WMO_INST_TYPE', 'Unknown')
        }
        
        return platform_data

    def _extract_profile_data(self, dataset: xr.Dataset, profile_idx: int, float_id: str) -> Dict[str, Any]:
        """Extract profile data for a specific profile index"""
        try:
            # Basic profile information
            profile_data = {
                'float_id': float_id,
                'cycle_number': int(dataset.CYCLE_NUMBER.values[profile_idx]) if hasattr(dataset, 'CYCLE_NUMBER') else profile_idx + 1,
                'profile_date': self._parse_juld_date(dataset.JULD.values[profile_idx], getattr(dataset, 'JULD_REFERENCE', '1950-01-01')),
                'latitude': float(dataset.LATITUDE.values[profile_idx]),
                'longitude': float(dataset.LONGITUDE.values[profile_idx]),
                'data_mode': getattr(dataset, 'DATA_MODE', 'R'),
                'source_file': str(dataset.encoding['source']) if 'source' in dataset.encoding else 'unknown'
            }
            
            # Extract parameter data with quality control
            parameters_data = self._extract_parameter_data(dataset, profile_idx)
            profile_data.update(parameters_data)
            
            # Calculate derived metrics
            profile_data.update(self._calculate_profile_metrics(parameters_data))
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error extracting profile {profile_idx}: {e}")
            return None


    def _extract_parameter_data(self, dataset: xr.Dataset, profile_idx: int) -> Dict[str, Any]:
        """Extract scientific parameter data with quality control"""
        parameters = {}
        
        # Pressure data (essential for vertical profiling)
        if 'PRES' in dataset.variables:
            pres_data = dataset.PRES.values[profile_idx]
            pres_qc = dataset.PRES_QC.values[profile_idx] if 'PRES_QC' in dataset.variables else '1' * len(pres_data)
            parameters['pressure_levels'] = self._apply_qc_filter(pres_data, pres_qc)
        
        # Temperature data
        if 'TEMP' in dataset.variables:
            temp_data = dataset.TEMP.values[profile_idx]
            temp_qc = dataset.TEMP_QC.values[profile_idx] if 'TEMP_QC' in dataset.variables else '1' * len(temp_data)
            parameters['temperature_values'] = self._apply_qc_filter(temp_data, temp_qc)
        
        # Salinity data
        if 'PSAL' in dataset.variables:
            psal_data = dataset.PSAL.values[profile_idx]
            psal_qc = dataset.PSAL_QC.values[profile_idx] if 'PSAL_QC' in dataset.variables else '1' * len(psal_data)
            parameters['salinity_values'] = self._apply_qc_filter(psal_data, psal_qc)
        
        # Oxygen data
        if 'DOXY' in dataset.variables:
            doxy_data = dataset.DOXY.values[profile_idx]
            doxy_qc = dataset.DOXY_QC.values[profile_idx] if 'DOXY_QC' in dataset.variables else '1' * len(doxy_data)
            parameters['oxygen_values'] = self._apply_qc_filter(doxy_data, doxy_qc)
        
        # Chlorophyll data
        if 'CHLA' in dataset.variables:
            chla_data = dataset.CHLA.values[profile_idx]
            chla_qc = dataset.CHLA_QC.values[profile_idx] if 'CHLA_QC' in dataset.variables else '1' * len(chla_data)
            parameters['chlorophyll_values'] = self._apply_qc_filter(chla_data, chla_qc)
        
        # Nitrate data
        if 'NITRATE' in dataset.variables:
            nitrate_data = dataset.NITRATE.values[profile_idx]
            nitrate_qc = dataset.NITRATE_QC.values[profile_idx] if 'NITRATE_QC' in dataset.variables else '1' * len(nitrate_data)
            parameters['nitrate_values'] = self._apply_qc_filter(nitrate_data, nitrate_qc)
        
        return parameters

    def _apply_qc_filter(self, data: np.ndarray, qc_flags: str, valid_flags: str = '123') -> List[float]:
        """Apply quality control filtering to data"""
        try:
            if data is None or len(data) == 0:
                return []
            
            filtered_data = []
            for i, value in enumerate(data):
                # Handle NaN values and QC flags
                if not np.isnan(value) and (i < len(qc_flags) and qc_flags[i] in valid_flags):
                    filtered_data.append(float(value))
            
            return filtered_data
        except Exception as e:
            logger.warning(f"Error applying QC filter: {e}")
            return [float(v) for v in data if not np.isnan(v)] if data is not None else []

    def _calculate_profile_metrics(self, parameters: Dict) -> Dict[str, Any]:
        """Calculate derived metrics for the profile"""
        metrics = {}
        
        # Basic statistics for each parameter
        for param_name, values in parameters.items():
            if values and len(values) > 0:
                clean_values = [v for v in values if v is not None and not np.isnan(v)]
                if clean_values:
                    metrics[f'{param_name}_min'] = float(np.min(clean_values))
                    metrics[f'{param_name}_max'] = float(np.max(clean_values))
                    metrics[f'{param_name}_mean'] = float(np.mean(clean_values))
                    metrics[f'{param_name}_std'] = float(np.std(clean_values))
        
        # Depth-related metrics
        if 'pressure_levels' in parameters and parameters['pressure_levels']:
            pressures = parameters['pressure_levels']
            if pressures:
                metrics['max_pressure'] = float(np.max(pressures))
                metrics['num_measurements'] = len(pressures)
                metrics['sampling_density'] = len(pressures) / (np.max(pressures) - np.min(pressures)) if np.max(pressures) > np.min(pressures) else 0
        
        return metrics

    def _get_platform_number(self, dataset: xr.Dataset) -> str:
        """Extract and format platform number"""
        try:
            if hasattr(dataset, 'PLATFORM_NUMBER'):
                platform_num = str(dataset.PLATFORM_NUMBER.values[0]).strip()
                # Clean up platform number formatting
                platform_num = platform_num.replace(' ', '')
                return platform_num
            else:
                return 'unknown'
        except:
            return 'unknown'

    def _extract_sensor_types(self, dataset: xr.Dataset) -> List[str]:
        """Extract sensor types from dataset"""
        sensors = []
        sensor_params = {
            'TEMP': 'Temperature Sensor',
            'PSAL': 'Salinity Sensor', 
            'PRES': 'Pressure Sensor',
            'DOXY': 'Oxygen Sensor',
            'CHLA': 'Fluorometer',
            'NITRATE': 'Nitrate Sensor'
        }
        
        for param, sensor_name in sensor_params.items():
            if param in dataset.variables:
                sensors.append(sensor_name)
        
        return sensors

    def _calculate_max_depth(self, dataset: xr.Dataset) -> float:
        """Calculate maximum depth from pressure data"""
        try:
            if 'PRES' in dataset.variables:
                return float(np.nanmax(dataset.PRES.values))
            return 2000.0  # Default ARGO float depth
        except:
            return 2000.0

    def _parse_juld_date(self, juld: float, reference_date: str) -> datetime:
        """Parse Julian date to datetime object"""
        try:
            # ARGO uses days since reference date (typically 1950-01-01)
            ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
            return ref_date + pd.Timedelta(days=float(juld))
        except:
            return datetime.utcnow()

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        try:
            if isinstance(date_str, datetime):
                return date_str
            for fmt in ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return datetime.utcnow()
        except:
            return datetime.utcnow()

    def download_argo_data(self, float_ids: List[str], data_dir: str, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Download ARGO data for specific floats"""
        results = {
            'downloaded_files': [],
            'errors': [],
            'total_floats': len(float_ids),
            'successful_downloads': 0
        }
        
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        for float_id in float_ids:
            try:
                logger.info(f"Downloading data for float {float_id}")
                downloaded_files = self._download_float_data(float_id, data_path, start_date, end_date)
                results['downloaded_files'].extend(downloaded_files)
                results['successful_downloads'] += 1
                
            except Exception as e:
                error_msg = f"Error downloading data for float {float_id}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        return results

    def _download_float_data(self, float_id: str, data_path: Path, 
                           start_date: Optional[datetime], end_date: Optional[datetime]) -> List[str]:
        """Download data for a single float"""
        # This is a simplified implementation - in practice you would use:
        # 1. ERDDAP servers
        # 2. GDAC FTP servers  
        # 3. Argo API endpoints
        
        downloaded_files = []
        
        try:
            # Example implementation using mock data for demonstration
            # In production, replace with actual ARGO data download logic
            logger.info(f"Mock download for float {float_id}")
            
            # Create mock NetCDF file for demonstration
            mock_file = self._create_mock_argo_file(float_id, data_path)
            if mock_file:
                downloaded_files.append(str(mock_file))
            
        except Exception as e:
            logger.error(f"Download failed for float {float_id}: {e}")
            raise
        
        return downloaded_files

    def _create_mock_argo_file(self, float_id: str, data_path: Path) -> Optional[Path]:
        """Create mock ARGO NetCDF file for testing/demonstration"""
        try:
            # Create a simple xarray Dataset mimicking ARGO data structure
            n_levels = 50
            n_profiles = 1
            
            ds = xr.Dataset({
                'PLATFORM_NUMBER': (['N_PROF'], [float_id]),
                'CYCLE_NUMBER': (['N_PROF'], [1]),
                'JULD': (['N_PROF'], [0.0]),
                'LATITUDE': (['N_PROF'], [45.0]),
                'LONGITUDE': (['N_PROF'], [-45.0]),
                'PRES': (['N_PROF', 'N_LEVELS'], [np.linspace(0, 500, n_levels)]),
                'TEMP': (['N_PROF', 'N_LEVELS'], [10 + 10 * np.exp(-np.linspace(0, 500, n_levels)/100)]),
                'PSAL': (['N_PROF', 'N_LEVELS'], [35.0 + 0.1 * np.sin(np.linspace(0, 500, n_levels)/100)]),
            })
            
            # Add global attributes
            ds.attrs = {
                'DATA_TYPE': 'ARGO PROFILE',
                'FORMAT_VERSION': '3.1',
                'DATA_CENTRE': 'CORIOLIS',
                'DATE_CREATION': datetime.utcnow().strftime('%Y%m%d%H%M%S'),
                'DATA_MODE': 'R'
            }
            
            # Save to file
            output_file = data_path / f"{float_id}_D000000001.nc"
            ds.to_netcdf(output_file)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error creating mock file: {e}")
            return None

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed data"""
        try:
            with db_manager.get_session() as session:
                # Float statistics
                total_floats = session.query(ArgoFloat).count()
                active_floats = session.query(ArgoFloat).filter(ArgoFloat.is_active == True).count()
                
                # Profile statistics
                total_profiles = session.query(ArgoProfile).count()
                profiles_by_mode = session.query(
                    ArgoProfile.data_mode, 
                    db_manager.func.count(ArgoProfile.id)
                ).group_by(ArgoProfile.data_mode).all()
                
                # Quality statistics
                avg_quality = session.query(db_manager.func.avg(ArgoProfile.quality_score)).scalar()
                
                return {
                    'total_floats': total_floats,
                    'active_floats': active_floats,
                    'total_profiles': total_profiles,
                    'profiles_by_mode': dict(profiles_by_mode),
                    'average_quality_score': float(avg_quality) if avg_quality else 0.0,
                    'last_processed_files': len(self.processed_files)
                }
                
        except Exception as e:
            logger.error(f"Error getting processing statistics: {e}")
            return {}

    def cleanup_old_data(self, older_than_days: int = 365) -> Dict[str, Any]:
        """Clean up old data from database"""
        try:
            cutoff_date = datetime.utcnow() - pd.Timedelta(days=older_than_days)
            
            with db_manager.get_session() as session:
                # Delete old profiles
                old_profiles = session.query(ArgoProfile).filter(
                    ArgoProfile.profile_date < cutoff_date
                ).delete()
                
                # Update floats that have no profiles
                session.query(ArgoFloat).filter(
                    ~ArgoFloat.float_id.in_(
                        session.query(ArgoProfile.float_id).distinct()
                    )
                ).update({'is_active': False})
                
                session.commit()
                
                return {
                    'deleted_profiles': old_profiles,
                    'cutoff_date': cutoff_date,
                    'status': 'success'
                }
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {'status': 'error', 'error': str(e)}

    def export_processed_data(self, output_format: str = 'csv', output_dir: str = None) -> Dict[str, Any]:
        """Export processed data to various formats"""
        try:
            if output_dir is None:
                output_dir = tempfile.mkdtemp()
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with db_manager.get_session() as session:
                # Export profiles
                profiles = session.query(ArgoProfile).all()
                profiles_data = []
                
                for profile in profiles:
                    profile_dict = {
                        'float_id': profile.float_id,
                        'cycle_number': profile.cycle_number,
                        'profile_date': profile.profile_date.isoformat(),
                        'latitude': profile.latitude,
                        'longitude': profile.longitude,
                        'quality_score': profile.quality_score,
                        'data_mode': profile.data_mode
                    }
                    profiles_data.append(profile_dict)
                
                df_profiles = pd.DataFrame(profiles_data)
                
                if output_format.lower() == 'csv':
                    output_file = output_path / 'argo_profiles.csv'
                    df_profiles.to_csv(output_file, index=False)
                elif output_format.lower() == 'parquet':
                    output_file = output_path / 'argo_profiles.parquet'
                    df_profiles.to_parquet(output_file, index=False)
                else:
                    raise ValueError(f"Unsupported format: {output_format}")
            
            return {
                'status': 'success',
                'output_file': str(output_file),
                'exported_profiles': len(profiles_data),
                'output_format': output_format
            }
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {'status': 'error', 'error': str(e)}

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate integrity of stored data"""
        validation_results = {
            'issues_found': [],
            'summary': {
                'total_checks': 0,
                'passed_checks': 0,
                'failed_checks': 0
            }
        }
        
        try:
            with db_manager.get_session() as session:
                # Check for profiles without corresponding floats
                orphaned_profiles = session.query(ArgoProfile).filter(
                    ~ArgoProfile.float_id.in_(session.query(ArgoFloat.float_id))
                ).count()
                
                if orphaned_profiles > 0:
                    validation_results['issues_found'].append(
                        f"Found {orphaned_profiles} profiles without corresponding floats"
                    )
                    validation_results['summary']['failed_checks'] += 1
                else:
                    validation_results['summary']['passed_checks'] += 1
                validation_results['summary']['total_checks'] += 1
                
                # Check for invalid coordinates
                invalid_coords = session.query(ArgoProfile).filter(
                    (ArgoProfile.latitude < -90) | (ArgoProfile.latitude > 90) |
                    (ArgoProfile.longitude < -180) | (ArgoProfile.longitude > 180)
                ).count()
                
                if invalid_coords > 0:
                    validation_results['issues_found'].append(
                        f"Found {invalid_coords} profiles with invalid coordinates"
                    )
                    validation_results['summary']['failed_checks'] += 1
                else:
                    validation_results['summary']['passed_checks'] += 1
                validation_results['summary']['total_checks'] += 1
                
                # Check for duplicate profiles
                duplicate_profiles = session.query(
                    ArgoProfile.float_id,
                    ArgoProfile.cycle_number,
                    db_manager.func.count(ArgoProfile.id)
                ).group_by(
                    ArgoProfile.float_id,
                    ArgoProfile.cycle_number
                ).having(db_manager.func.count(ArgoProfile.id) > 1).all()
                
                if duplicate_profiles:
                    validation_results['issues_found'].append(
                        f"Found {len(duplicate_profiles)} sets of duplicate profiles"
                    )
                    validation_results['summary']['failed_checks'] += 1
                else:
                    validation_results['summary']['passed_checks'] += 1
                validation_results['summary']['total_checks'] += 1
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
            validation_results['issues_found'].append(f"Validation error: {str(e)}")
            return validation_results

# Utility function for batch processing
def process_argo_batch(netcdf_dirs: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Batch process multiple ARGO data directories"""
    if config is None:
        config = {}
    
    ingestor = ArgoDataIngestor()
    overall_results = {
        'total_directories': len(netcdf_dirs),
        'directory_results': [],
        'start_time': datetime.utcnow(),
        'end_time': None
    }
    
    for netcdf_dir in netcdf_dirs:
        try:
            logger.info(f"Processing directory: {netcdf_dir}")
            dir_results = ingestor.process_netcdf_directory(
                netcdf_dir, 
                force_reprocess=config.get('force_reprocess', False)
            )
            overall_results['directory_results'].append({
                'directory': netcdf_dir,
                'results': dir_results
            })
        except Exception as e:
            logger.error(f"Error processing directory {netcdf_dir}: {e}")
            overall_results['directory_results'].append({
                'directory': netcdf_dir,
                'error': str(e)
            })
    
    overall_results['end_time'] = datetime.utcnow()
    overall_results['processing_duration'] = (
        overall_results['end_time'] - overall_results['start_time']
    ).total_seconds()
    
    return overall_results
    