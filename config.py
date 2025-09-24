# config.py
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for FloatChat"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://argouser:argopass@localhost:5432/argodb")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "vector_db"))
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # ARGO Data Sources
    ARGO_FTP_SERVER = os.getenv("ARGO_FTP_SERVER", "ftp.ifremer.fr")
    INCOIS_API_BASE = os.getenv("INCOIS_API_BASE", "https://incois.gov.in/OON")
    
    # Application
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        return {
            "url": cls.DATABASE_URL,
            "pool_size": 20,
            "max_overflow": 30,
            "echo": cls.DEBUG
        }
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed",
            cls.DATA_DIR / "exports",
            cls.LOGS_DIR,
            Path(cls.VECTOR_DB_PATH)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
Config.ensure_directories()