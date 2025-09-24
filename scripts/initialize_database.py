#!/usr/bin/env python3
# scripts/initialize_database.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.database_manager import db_manager
from src.config import config
import logging
import argparse

def setup_logging():
    """Setup logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/database_setup.log')
        ]
    )

def initialize_database(force_recreate: bool = False, backup_first: bool = True) -> bool:
    """Initialize database with enhanced features"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting database initialization...")
        
        # Load configuration
        config.load_config()
        
        # Backup existing database if requested
        if backup_first and not force_recreate:
            db_path = Path("argo_chat.db")
            if db_path.exists():
                backup_path = Path("backups")
                backup_path.mkdir(exist_ok=True)
                db_manager.backup_database(str(backup_path))
        
        # Initialize database connection
        db_url = config.get('database.url')
        if not db_manager.initialize_database(db_url):
            logger.error("Database connection initialization failed")
            return False
        
        # Create tables
        db_manager.create_tables()
        logger.info("Database tables created successfully")
        
        # Run initial optimizations
        db_manager.optimize_database()
        logger.info("Database optimization completed")
        
        # Display database statistics
        stats = db_manager.get_database_stats()
        logger.info(f"Database statistics: {stats}")
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def main():
    """Main function for database initialization script"""
    parser = argparse.ArgumentParser(description='Initialize ARGO AI Database')
    parser.add_argument('--force', action='store_true', 
                       help='Force recreation of database')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backup of existing database')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load custom config if provided
    if args.config:
        config.load_config(args.config)
    
    # Initialize database
    success = initialize_database(
        force_recreate=args.force,
        backup_first=not args.no_backup
    )
    
    if success:
        logger.info("Database setup completed successfully")
        return 0
    else:
        logger.error("Database setup failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())