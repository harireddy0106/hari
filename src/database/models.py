# src/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
import datetime
import enum

class DataMode(enum.Enum):
    REAL_TIME = "R"
    ADJUSTED = "A" 
    DELAYED = "D"

Base = declarative_base()

class ArgoFloat(Base):
    __tablename__ = "argo_floats"
    
    id = Column(Integer, primary_key=True, index=True)
    float_id = Column(String(50), unique=True, index=True, nullable=False)
    platform_number = Column(String(20), nullable=False)
    data_center = Column(String(100))
    institution = Column(String(200))
    date_created = Column(DateTime)
    date_updated = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Location info
    last_latitude = Column(Float)
    last_longitude = Column(Float)
    last_transmission = Column(DateTime)
    
    # Technical specs
    sensor_types = Column(JSON)
    max_depth = Column(Integer)
    cycle_count = Column(Integer, default=0)
    
    # Quality control fields
    data_mode = Column(String(1), default="R")  # R: Real-time, A: Adjusted, D: Delayed
    quality_score = Column(Float, default=100.0)
    qc_tests_applied = Column(JSON)

class ArgoProfile(Base):
    __tablename__ = "argo_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    float_id = Column(String(50), index=True, nullable=False)
    cycle_number = Column(Integer, nullable=False)
    profile_date = Column(DateTime, nullable=False)
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Measurements with QC flags
    pressure_levels = Column(JSON)
    pressure_qc = Column(JSON)
    
    temperature_values = Column(JSON)
    temperature_qc = Column(JSON)
    
    salinity_values = Column(JSON)
    salinity_qc = Column(JSON)
    
    # BGC parameters
    oxygen_values = Column(JSON, nullable=True)
    chlorophyll_values = Column(JSON, nullable=True)
    nitrate_values = Column(JSON, nullable=True)
    ph_values = Column(JSON, nullable=True)
    
    # Quality control
    data_mode = Column(String(1), default="R")
    quality_score = Column(Float, default=100.0)
    qc_comments = Column(Text)
    
    # File reference
    source_file = Column(String(500))
    file_format = Column(String(10))

class UserQuery(Base):
    __tablename__ = "user_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True)
    natural_language_query = Column(Text, nullable=False)
    generated_sql = Column(Text)
    query_results_count = Column(Integer)
    response_text = Column(Text)
    visualization_type = Column(String(50))
    
    # Performance metrics
    processing_time_ms = Column(Integer)
    query_complexity = Column(String(20))  # simple, medium, complex
    
    # Timestamps
    query_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # User info (anonymous)
    user_agent = Column(String(500))
    ip_address = Column(String(45))

class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    content_type = Column(String(50))
    content_id = Column(String(100), index=True)
    embedding_vector = Column(JSON)  # Store as JSON instead of ARRAY
    content_text = Column(Text)
    embedding_metadata = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)