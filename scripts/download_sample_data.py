#!/usr/bin/env python3
# scripts/download_sample_data.py

import requests
import json
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta
import tempfile
import zipfile
from urllib.parse import urljoin

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from database.database_manager import db_manager
from data.argo_data_ingestor import ArgoDataIngestor
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/download_sample_data.log')
    ]
)
logger = logging.getLogger(__name__)

class ArgoSampleDataDownloader:
    """Enhanced sample data downloader with multiple sources"""
    
    def __init__(self):
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # ARGO data sources
        self.data_sources = {
            'gdac': {
                'base_url': 'https://data-argo.ifremer.fr/',
                'profiles_path': 'dac/coriolis/'
            },
            'erddap': {
                'base_url': 'https://erddap.ifremer.fr/erddap/',
                'dataset': 'ArgoFloats'
            },
            'demo': {
                'base_url': 'https://argovis.colorado.edu/',
                'api_path': 'api/v1/profiles'
            }
        }
        
        # Sample float IDs for different regions
        self.sample_floats = {
            'indian_ocean': ['1902256', '1902257', '2902256'],
            'atlantic': ['6902743', '6902744', '6902745'],
            'pacific': ['5904449', '5904450', '5904451'],
            'southern_ocean': ['2902258', '2902259', '2902260']
        }
    
    def download_from_erddap(self, float_ids: list, region: str = 'indian_ocean') -> list:
        """Download data from ERDDAP server"""
        downloaded_files = []
        
        try:
            base_url = self.data_sources['erddap']['base_url']
            dataset = self.data_sources['erddap']['dataset']
            
            for float_id in float_ids[:3]:  # Limit to 3 floats per region
                try:
                    # Construct ERDDAP query
                    query_params = {
                        'platform_number': float_id,
                        'format': 'nc',
                        'start_date': '2020-01-01',
                        'end_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    query_url = f"{base_url}tabledap/{dataset}.nc"
                    response = requests.get(query_url, params=query_params, stream=True, timeout=30)
                    
                    if response.status_code == 200:
                        filename = f"argo_{float_id}_{region}.nc"
                        file_path = self.data_dir / filename
                        
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        downloaded_files.append(file_path)
                        logger.info(f"Downloaded data for float {float_id} from {region}")
                    
                    else:
                        logger.warning(f"Failed to download data for float {float_id}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"Error downloading float {float_id}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"ERDDAP download error: {e}")
        
        return downloaded_files
    
    def download_demo_data(self) -> list:
        """Download demo data package"""
        demo_files = []
        
        try:
            # Create sample NetCDF files for demonstration
            sample_data = self._create_sample_netcdf_files()
            demo_files.extend(sample_data)
            
            logger.info("Created demo NetCDF files")
            
        except Exception as e:
            logger.error(f"Demo data creation error: {e}")
        
        return demo_files
    
    def _create_sample_netcdf_files(self) -> list:
        """Create sample NetCDF files for demonstration"""
        import xarray as xr
        import numpy as np
        
        created_files = []
        
        # Create sample data for different regions
        regions = {
            'indian_ocean': {'lat_range': (10, 25), 'lon_range': (60, 85)},
            'arabian_sea': {'lat_range': (15, 25), 'lon_range': (55, 75)},
            'bay_of_bengal': {'lat_range': (10, 20), 'lon_range': (80, 95)}
        }
        
        for region_name, coords in regions.items():
            for i in range(2):  # Create 2 files per region
                try:
                    # Create sample dataset
                    n_profiles = 5
                    n_levels = 50
                    
                    # Generate realistic data
                    latitudes = np.random.uniform(coords['lat_range'][0], coords['lat_range'][1], n_profiles)
                    longitudes = np.random.uniform(coords['lon_range'][0], coords['lon_range'][1], n_profiles)
                    
                    # Create xarray dataset
                    ds = xr.Dataset({
                        'LATITUDE': (['N_PROF'], latitudes),
                        'LONGITUDE': (['N_PROF'], longitudes),
                        'JULD': (['N_PROF'], np.arange(n_profiles) * 10),
                        'PRES': (['N_PROF', 'N_LEVELS'], 
                                np.random.uniform(0, 500, (n_profiles, n_levels))),
                        'TEMP': (['N_PROF', 'N_LEVELS'], 
                                10 + 10 * np.exp(-np.random.uniform(0, 500, (n_profiles, n_levels))/100)),
                        'PSAL': (['N_PROF', 'N_LEVELS'], 
                                35.0 + 0.1 * np.sin(np.random.uniform(0, 500, (n_profiles, n_levels))/100)),
                    })
                    
                    # Add global attributes
                    ds.attrs = {
                        'DATA_TYPE': 'ARGO PROFILE',
                        'FORMAT_VERSION': '3.1',
                        'DATA_CENTRE': 'CORIOLIS',
                        'PLATFORM_NUMBER': f'190{region_name[:3].upper()}{i}',
                        'DATE_CREATION': datetime.now().strftime('%Y%m%d%H%M%S')
                    }
                    
                    # Save to file
                    filename = f"sample_{region_name}_{i+1}.nc"
                    file_path = self.data_dir / filename
                    ds.to_netcdf(file_path)
                    
                    created_files.append(file_path)
                    logger.info(f"Created sample file: {filename}")
                
                except Exception as e:
                    logger.error(f"Error creating sample file for {region_name}: {e}")
        
        return created_files
    
    def process_downloaded_data(self, file_paths: list) -> dict:
        """Process downloaded NetCDF files"""
        ingestor = ArgoDataIngestor()
        results = {
            'processed_files': 0,
            'successful_files': 0,
            'errors': []
        }
        
        for file_path in file_paths:
            try:
                file_results = ingestor.process_netcdf_file(file_path)
                if file_results:
                    ingestor.store_netcdf_data(file_results, str(file_path), 
                                             ingestor._calculate_file_hash(file_path))
                    results['successful_files'] += 1
                
                results['processed_files'] += 1
                logger.info(f"Processed {file_path.name}")
            
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        return results
    
    def download_comprehensive_sample(self, regions: list = None) -> dict:
        """Download comprehensive sample dataset"""
        if regions is None:
            regions = ['indian_ocean', 'atlantic', 'pacific']
        
        all_downloaded = []
        results = {}
        
        logger.info(f"Starting comprehensive download for regions: {regions}")
        
        for region in regions:
            if region in self.sample_floats:
                logger.info(f"Downloading data for {region}")
                
                # Try ERDDAP first
                region_files = self.download_from_erddap(self.sample_floats[region], region)
                
                if not region_files:
                    logger.warning(f"No data downloaded from ERDDAP for {region}, using demo data")
                    region_files = self.download_demo_data()
                
                all_downloaded.extend(region_files)
                results[region] = {
                    'downloaded_files': len(region_files),
                    'file_paths': [str(f) for f in region_files]
                }
        
        # Process all downloaded files
        if all_downloaded:
            processing_results = self.process_downloaded_data(all_downloaded)
            results['processing'] = processing_results
            
            logger.info(f"Download and processing completed: {processing_results['successful_files']} files processed")
        else:
            logger.warning("No files were downloaded")
            results['processing'] = {'successful_files': 0, 'errors': ['No files downloaded']}
        
        return results

def main():
    """Main function for sample data download"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download ARGO sample data')
    parser.add_argument('--regions', nargs='+', 
                       choices=['indian_ocean', 'atlantic', 'pacific', 'southern_ocean', 'all'],
                       default=['indian_ocean'],
                       help='Regions to download data for')
    parser.add_argument('--demo-only', action='store_true',
                       help='Use only demo data (no internet download)')
    parser.add_argument('--process', action='store_true', default=True,
                       help='Process downloaded data into database')
    
    args = parser.parse_args()
    
    # Initialize database
    config.load_config()
    if not db_manager.initialize_database(config.get('database.url')):
        logger.error("Database initialization failed")
        return 1
    
    # Create tables if they don't exist
    db_manager.create_tables()
    
    # Download data
    downloader = ArgoSampleDataDownloader()
    
    if 'all' in args.regions:
        regions = ['indian_ocean', 'atlantic', 'pacific', 'southern_ocean']
    else:
        regions = args.regions
    
    try:
        if args.demo_only:
            logger.info("Using demo data only")
            results = downloader.download_demo_data()
        else:
            logger.info(f"Downloading data for regions: {regions}")
            results = downloader.download_comprehensive_sample(regions)
        
        # Display summary
        logger.info("Download summary:")
        for region, info in results.items():
            if region != 'processing':
                logger.info(f"  {region}: {info['downloaded_files']} files")
        
        if 'processing' in results:
            proc = results['processing']
            logger.info(f"Processing: {proc['successful_files']} successful, {len(proc['errors'])} errors")
        
        return 0
    
    except Exception as e:
        logger.error(f"Sample data download failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())