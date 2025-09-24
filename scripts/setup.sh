# scripts/download_sample_data.py
import requests
import os
import zipfile
from pathlib import Path
import logging
from tqdm import tqdm
import tempfile
from src.config import config

class SampleDataDownloader:
    """Download sample Argo data for testing and demonstration"""
    
    SAMPLE_DATA_SOURCES = {
        'atlantic_2023': {
            'url': 'https://data-argo.ifremer.fr/argo/dac/aoml/1902256/profiles/',
            'files': ['R1902256_001.nc', 'R1902256_002.nc'],
            'description': 'Float 1902256 in Atlantic Ocean (2023)'
        },
        'pacific_2023': {
            'url': 'https://data-argo.ifremer.fr/argo/dac/coriolis/2903754/profiles/',
            'files': ['R2903754_001.nc', 'R2903754_002.nc'],
            'description': 'Float 2903754 in Pacific Ocean (2023)'
        },
        'indian_2023': {
            'url': 'https://data-argo.ifremer.fr/argo/dac/jma/4902677/profiles/',
            'files': ['R4902677_001.nc', 'R4902677_002.nc'],
            'description': 'Float 4902677 in Indian Ocean (2023)'
        }
    }
    
    def __init__(self):
        self.raw_dir = config.get_data_dir('raw')
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/download_sample_data.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a single file with progress bar"""
        try:
            local_path = self.raw_dir / filename
            
            # Skip if file already exists
            if local_path.exists():
                self.logger.info(f"File already exists: {filename}")
                return True
            
            full_url = f"{url}{filename}"
            self.logger.info(f"Downloading: {full_url}")
            
            # Stream download with progress bar
            response = requests.get(full_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(local_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
            
            self.logger.info(f"Successfully downloaded: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def download_sample_dataset(self, dataset_key: str = 'atlantic_2023') -> bool:
        """Download a specific sample dataset"""
        if dataset_key not in self.SAMPLE_DATA_SOURCES:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        dataset = self.SAMPLE_DATA_SOURCES[dataset_key]
        self.logger.info(f"Downloading dataset: {dataset['description']}")
        
        success_count = 0
        for filename in dataset['files']:
            if self.download_file(dataset['url'], filename):
                success_count += 1
        
        self.logger.info(f"Downloaded {success_count}/{len(dataset['files'])} files")
        return success_count == len(dataset['files'])
    
    def download_all_samples(self) -> bool:
        """Download all available sample datasets"""
        self.logger.info("Starting download of all sample datasets")
        
        success_count = 0
        for dataset_key in self.SAMPLE_DATA_SOURCES.keys():
            if self.download_sample_dataset(dataset_key):
                success_count += 1
        
        total_datasets = len(self.SAMPLE_DATA_SOURCES)
        self.logger.info(f"Downloaded {success_count}/{total_datasets} datasets")
        return success_count == total_datasets
    
    def validate_downloaded_files(self) -> dict:
        """Validate downloaded files by checking their integrity"""
        validation_results = {}
        
        for file_path in self.raw_dir.glob("*.nc"):
            try:
                file_size = file_path.stat().st_size
                is_valid = file_size > 1000  # Basic size check
                
                validation_results[file_path.name] = {
                    'size_kb': file_size / 1024,
                    'is_valid': is_valid,
                    'error': None if is_valid else "File too small or corrupted"
                }
                
                if is_valid:
                    self.logger.info(f"✓ {file_path.name} ({file_size/1024:.1f} KB)")
                else:
                    self.logger.warning(f"✗ {file_path.name} - possible corruption")
                    
            except Exception as e:
                validation_results[file_path.name] = {
                    'size_kb': 0,
                    'is_valid': False,
                    'error': str(e)
                }
                self.logger.error(f"Error validating {file_path.name}: {e}")
        
        return validation_results
    
    def create_sample_metadata(self):
        """Create metadata file for downloaded samples"""
        metadata = {
            'download_date': str(Path(__file__).parent.parent),
            'datasets': self.SAMPLE_DATA_SOURCES,
            'total_files': len(list(self.raw_dir.glob("*.nc")))
        }
        
        metadata_file = self.raw_dir / "sample_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Sample metadata saved to: {metadata_file}")

def main():
    """Main function to download sample data"""
    downloader = SampleDataDownloader()
    
    print("🌊 ArgoChat Sample Data Downloader")
    print("=" * 40)
    
    # Create raw directory if it doesn't exist
    downloader.raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download options
    print("\nAvailable datasets:")
    for key, info in downloader.SAMPLE_DATA_SOURCES.items():
        print(f"  {key}: {info['description']}")
    
    choice = input("\nDownload (a)ll datasets or (s)pecific dataset? [a/s]: ").lower()
    
    success = False
    if choice == 's':
        dataset_key = input("Enter dataset key: ").strip()
        success = downloader.download_sample_dataset(dataset_key)
    else:
        success = downloader.download_all_samples()
    
    if success:
        # Validate downloads
        print("\nValidating downloaded files...")
        results = downloader.validate_downloaded_files()
        
        valid_files = sum(1 for r in results.values() if r['is_valid'])
        print(f"Valid files: {valid_files}/{len(results)}")
        
        # Create metadata
        downloader.create_sample_metadata()
        
        print("\n✅ Sample data download completed successfully!")
        print(f"Files saved to: {downloader.raw_dir}")
    else:
        print("\n❌ Sample data download failed. Check logs for details.")

if __name__ == "__main__":
    main()