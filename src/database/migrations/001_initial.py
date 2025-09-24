# src/database/migrations/001_initial.py
from sqlalchemy import text
from database.database_manager import db_manager
import logging

def run_migration():
    """Run initial database migration"""
    
    migration_scripts = [
        # Create argo_profiles table
        """
        CREATE TABLE IF NOT EXISTS argo_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            float_id TEXT NOT NULL,
            cycle_number INTEGER NOT NULL,
            profile_date DATETIME NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            quality_score REAL DEFAULT 1.0,
            pressure_levels TEXT,  -- JSON array of pressures
            temperature_values TEXT,  -- JSON array of temperatures
            salinity_values TEXT,  -- JSON array of salinities
            oxygen_values TEXT,  -- JSON array of oxygen values
            chlorophyll_values TEXT,  -- JSON array of chlorophyll values
            source_file TEXT,
            metadata TEXT,  -- JSON string for additional metadata
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(float_id, cycle_number)
        )
        """,
        
        # Create indexes for better performance
        """
        CREATE INDEX IF NOT EXISTS idx_argo_profiles_float_id 
        ON argo_profiles(float_id)
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_argo_profiles_date 
        ON argo_profiles(profile_date)
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_argo_profiles_location 
        ON argo_profiles(latitude, longitude)
        """,
        
        # Create file_metadata table for tracking processed files
        """
        CREATE TABLE IF NOT EXISTS file_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            profiles_count INTEGER DEFAULT 0,
            processing_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'processed',
            error_message TEXT
        )
        """,
        
        # Create user_queries table for NLP query logging
        """
        CREATE TABLE IF NOT EXISTS user_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            processed_query TEXT,
            results_count INTEGER DEFAULT 0,
            response_time REAL DEFAULT 0.0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT
        )
        """,
        
        # Create visualization_settings table
        """
        CREATE TABLE IF NOT EXISTS visualization_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT UNIQUE NOT NULL,
            setting_value TEXT NOT NULL,
            description TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    try:
        with db_manager.get_session() as session:
            # Check if migration has already been run
            result = session.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='argo_profiles'
            """)).fetchone()
            
            if result:
                logging.info("Database already initialized")
                return True
            
            # Run migration scripts
            for script in migration_scripts:
                session.execute(text(script))
            
            # Insert default visualization settings
            default_settings = [
                ('color_palette', 'Viridis', 'Default color palette for plots'),
                ('default_theme', 'plotly_white', 'Default plot theme'),
                ('max_profiles_display', '1000', 'Maximum profiles to display'),
                ('auto_refresh', 'true', 'Enable auto-refresh of data')
            ]
            
            for name, value, description in default_settings:
                session.execute(text("""
                    INSERT OR REPLACE INTO visualization_settings 
                    (setting_name, setting_value, description) 
                    VALUES (:name, :value, :desc)
                """), {'name': name, 'value': value, 'desc': description})
            
            session.commit()
            logging.info("Database migration completed successfully")
            return True
            
    except Exception as e:
        logging.error(f"Database migration failed: {e}")
        return False

def add_new_columns():
    """Add new columns to existing tables (for future migrations)"""
    
    new_columns = [
        ("argo_profiles", "water_mass", "TEXT"),
        ("argo_profiles", "season", "TEXT"),
        ("argo_profiles", "region", "TEXT")
    ]
    
    try:
        with db_manager.get_session() as session:
            for table, column, dtype in new_columns:
                # Check if column exists
                result = session.execute(text(f"""
                    PRAGMA table_info({table})
                """)).fetchall()
                
                columns = [col[1] for col in result]
                if column not in columns:
                    session.execute(text(f"""
                        ALTER TABLE {table} ADD COLUMN {column} {dtype}
                    """))
            
            session.commit()
            logging.info("New columns added successfully")
            return True
            
    except Exception as e:
        logging.error(f"Failed to add new columns: {e}")
        return False