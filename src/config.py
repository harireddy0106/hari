# src/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class Config:
    """Configuration manager for ArgoChat application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.settings = self._load_settings()
        self._setup_logging()
    
    def _find_config_file(self) -> Path:
        """Find the configuration file in various locations"""
        possible_paths = [
            Path('config/settings.yaml'),
            Path('../config/settings.yaml'),
            Path('src/config/settings.yaml'),
            Path('./settings.yaml')
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Create default config if not found
        default_config = Path('config/settings.yaml')
        default_config.parent.mkdir(exist_ok=True)
        self._create_default_config(default_config)
        return default_config
    
    def _create_default_config(self, config_path: Path):
        """Create default configuration file"""
        default_config = {
            'database': {
                'url': 'sqlite:///argo_chat.db',
                'echo': False,
                'pool_pre_ping': True
            },
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'exports_dir': 'data/exports',
                'vector_db_dir': 'data/vector_db',
                'max_file_size_mb': 100
            },
            'nlp': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'similarity_threshold': 0.7,
                'max_results': 50
            },
            'visualization': {
                'default_theme': 'plotly_white',
                'color_palette': 'Viridis',
                'max_profiles_display': 1000
            },
            'api': {
                'argo_data_url': 'https://data-argo.ifremer.fr',
                'timeout': 30,
                'retry_attempts': 3
            },
            'quality_control': {
                'temperature_range': [-2, 40],
                'salinity_range': [0, 42],
                'pressure_range': [0, 4000],
                'quality_threshold': 0.7
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.get('logging.level', 'INFO'))
        log_format = self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/argo_chat.log'),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        # Environment variable override
        env_key = f"ARGO_{key.replace('.', '_').upper()}"
        return os.getenv(env_key, value)
    
    def get_database_url(self) -> str:
        """Get database URL with environment variable override"""
        return os.getenv('DATABASE_URL', self.get('database.url'))
    
    def get_data_dir(self, dir_type: str) -> Path:
        """Get data directory path"""
        dir_path = Path(self.get(f'data.{dir_type}_dir'))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

# Global configuration instance
config = Config()