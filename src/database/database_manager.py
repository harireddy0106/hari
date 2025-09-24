# src/database/database_manager.py

import sqlalchemy as sa
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
import logging
from typing import Optional, Dict, Any
import os
from pathlib import Path
import json
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.engine = None
            self.SessionLocal = None
            self.Base = declarative_base()
            self._initialized = True
    
    def initialize_database(self, database_url: Optional[str] = None, echo: bool = False) -> bool:
        """Initialize database connection with enhanced features"""
        try:
            if database_url is None:
                # Default to SQLite with WAL mode for better concurrency
                db_path = Path("argo_chat.db")
                database_url = f"sqlite:///{db_path.absolute()}"
            
            # SQLite specific optimizations
            connect_args = {}
            if database_url.startswith('sqlite'):
                connect_args = {
                    'check_same_thread': False,
                    'timeout': 30  # 30 second timeout
                }
                
                # Enable WAL mode for better concurrency
                @event.listens_for(create_engine(database_url), 'connect')
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()
            
            # Create engine with optimizations
            self.engine = create_engine(
                database_url,
                echo=echo,
                connect_args=connect_args,
                poolclass=StaticPool if database_url.startswith('sqlite') else None,
                pool_pre_ping=True,  # Verify connection before use
                max_overflow=10,     # Allow 10 connections beyond pool_size
                pool_size=5          # Maintain 5 connections in pool
            )
            
            # Create session factory
            self.SessionLocal = scoped_session(
                sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine,
                    expire_on_commit=False  # Better for web applications
                )
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            
            logger.info(f"Database initialized successfully: {database_url}")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def get_session(self):
        """Get database session with error handling"""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize_database() first.")
        
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all tables with proper error handling"""
        try:
            from .models import ArgoFloat, ArgoProfile, UserQuery, SystemLog
            self.Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with self.engine.connect() as conn:
                # Table statistics
                stats = {}
                
                # Count records in each table
                tables = ['argo_floats', 'argo_profiles', 'user_queries', 'system_logs']
                for table in tables:
                    result = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
                    stats[f'{table}_count'] = result.scalar()
                
                # Database size
                if self.engine.url.drivername == 'sqlite':
                    db_path = Path("argo_chat.db")
                    if db_path.exists():
                        stats['database_size_mb'] = round(db_path.stat().st_size / (1024 * 1024), 2)
                
                # Last update time
                result = conn.execute(sa.text("SELECT MAX(profile_date) FROM argo_profiles"))
                stats['last_profile_date'] = result.scalar()
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def optimize_database(self) -> bool:
        """Perform database optimization tasks"""
        try:
            with self.engine.connect() as conn:
                if self.engine.url.drivername == 'sqlite':
                    # SQLite specific optimizations
                    conn.execute(sa.text("VACUUM"))
                    conn.execute(sa.text("PRAGMA optimize"))
                    conn.execute(sa.text("PRAGMA analysis_limit=400"))
                    logger.info("SQLite database optimized")
                
                # Update statistics for all tables
                conn.execute(sa.text("ANALYZE"))
                logger.info("Database statistics updated")
                
                return True
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            if self.engine.url.drivername == 'sqlite':
                import shutil
                db_path = Path("argo_chat.db")
                backup_file = Path(backup_path) / f"argo_chat_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                
                shutil.copy2(db_path, backup_file)
                logger.info(f"Database backed up to: {backup_file}")
                return True
            else:
                logger.warning("Backup currently only supported for SQLite")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def execute_raw_query(self, query: str, params: Dict = None) -> Any:
        """Execute raw SQL query with safety checks"""
        if params is None:
            params = {}
        
        # Basic safety check - prevent destructive operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        if any(keyword in query.upper() for keyword in dangerous_keywords):
            raise ValueError("Potentially dangerous query detected")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(query), params)
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row) for row in result.mappings()]
                else:
                    conn.commit()
                    return result.rowcount
                    
        except Exception as e:
            logger.error(f"Raw query execution failed: {e}")
            raise
    
    def close_connections(self):
        """Close all database connections"""
        try:
            if self.SessionLocal:
                self.SessionLocal.remove()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global database manager instance
db_manager = DatabaseManager()