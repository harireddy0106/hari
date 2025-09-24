# argo_ai_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import json
import requests
import xarray as xr
import tempfile
import os
from pathlib import Path
import sys
import io
import base64
from PIL import Image
import logging
import gzip
import zipfile
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="ARGO AI Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: 2rem;
        border-left: 4px solid #1f77b4;
    }
    .ai-message {
        background-color: #f0f0f0;
        margin-right: 2rem;
        border-left: 4px solid #ff6b6b;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    /* Fix map container */
    .folium-map {
        border-radius: 10px;
        border: 2px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class ArgoDataProcessor:
    """Process ARGO data files"""
    
    def __init__(self):
        self.processed_data = {}
        
    def process_netcdf_file(self, file_path):
        """Process NetCDF file and extract ARGO data"""
        try:
            with xr.open_dataset(file_path) as ds:
                data = {}
                
                # Extract basic information
                data['variables'] = list(ds.variables.keys())
                data['dimensions'] = dict(ds.dims)
                
                # Extract float data
                if 'LATITUDE' in ds.variables:
                    data['latitude'] = float(ds.LATITUDE.values)
                if 'LONGITUDE' in ds.variables:
                    data['longitude'] = float(ds.LONGITUDE.values)
                if 'JULD' in ds.variables:
                    data['date'] = ds.JULD.values
                
                # Extract profile data
                if 'TEMP' in ds.variables:
                    data['temperature'] = ds.TEMP.values
                if 'PSAL' in ds.variables:
                    data['salinity'] = ds.PSAL.values
                if 'PRES' in ds.variables:
                    data['pressure'] = ds.PRES.values
                
                return data
        except Exception as e:
            st.error(f"Error processing NetCDF file: {str(e)}")
            return None

class ArgoAIDashboard:
    def __init__(self):
        self.setup_logging()
        self.data_processor = ArgoDataProcessor()
        self.initialize_data()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_data(self):
        """Initialize sample data and uploaded data storage"""
        # Sample ARGO float data
        self.floats_data = self.generate_sample_floats()
        self.profiles_data = self.generate_sample_profiles()
        
        # Storage for uploaded data
        if 'uploaded_floats' not in st.session_state:
            st.session_state.uploaded_floats = pd.DataFrame()
        if 'uploaded_profiles' not in st.session_state:
            st.session_state.uploaded_profiles = pd.DataFrame()
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
    
    def get_combined_data(self):
        """Combine sample and uploaded data"""
        floats_combined = self.floats_data.copy()
        profiles_combined = self.profiles_data.copy()
        
        if not st.session_state.uploaded_floats.empty:
            floats_combined = pd.concat([floats_combined, st.session_state.uploaded_floats], ignore_index=True)
        if not st.session_state.uploaded_profiles.empty:
            profiles_combined = pd.concat([profiles_combined, st.session_state.uploaded_profiles], ignore_index=True)
            
        return floats_combined, profiles_combined
    
    def generate_sample_floats(self):
        """Generate sample ARGO float data for 5 oceans"""
        oceans = {
            'Pacific': {'lat_range': (-60, 60), 'lon_range': (120, -60)},
            'Atlantic': {'lat_range': (-60, 60), 'lon_range': (-80, 20)},
            'Indian': {'lat_range': (-60, 30), 'lon_range': (20, 120)},
            'Southern': {'lat_range': (-90, -60), 'lon_range': (-180, 180)},
            'Arctic': {'lat_range': (60, 90), 'lon_range': (-180, 180)}
        }
        
        floats = []
        float_id = 2900000
        
        for ocean, ranges in oceans.items():
            for i in range(20):  # 20 floats per ocean
                lat = np.random.uniform(ranges['lat_range'][0], ranges['lat_range'][1])
                lon = np.random.uniform(ranges['lon_range'][0], ranges['lon_range'][1])
                
                floats.append({
                    'float_id': float_id + i,
                    'ocean': ocean,
                    'latitude': lat,
                    'longitude': lon,
                    'status': np.random.choice(['Active', 'Inactive'], p=[0.8, 0.2]),
                    'last_transmission': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                    'profiles_count': np.random.randint(50, 500),
                    'data_quality': np.random.uniform(70, 95),
                    'data_source': 'Sample'
                })
            float_id += 1000
            
        return pd.DataFrame(floats)
    
    def generate_sample_profiles(self):
        """Generate sample profile data"""
        profiles = []
        profile_id = 10000
        
        for _, float_data in self.floats_data.iterrows():
            for cycle in range(1, min(10, float_data['profiles_count'] // 50)):
                # Generate pressure levels (0-2000 dbar)
                pressure_levels = np.linspace(0, 2000, 50)
                
                # Generate temperature profile (typical ocean profile)
                surface_temp = np.random.uniform(-2, 30)
                deep_temp = np.random.uniform(-2, 4)
                temp_profile = self.generate_temperature_profile(pressure_levels, surface_temp, deep_temp)
                
                # Generate salinity profile
                salinity_profile = self.generate_salinity_profile(pressure_levels)
                
                profiles.append({
                    'profile_id': profile_id,
                    'float_id': float_data['float_id'],
                    'cycle_number': cycle,
                    'date': float_data['last_transmission'] - timedelta(days=cycle*10),
                    'latitude': float_data['latitude'] + np.random.uniform(-0.5, 0.5),
                    'longitude': float_data['longitude'] + np.random.uniform(-0.5, 0.5),
                    'pressure_levels': pressure_levels,
                    'temperature': temp_profile,
                    'salinity': salinity_profile,
                    'quality_score': np.random.uniform(80, 95),
                    'data_source': 'Sample'
                })
                profile_id += 1
                
        return pd.DataFrame(profiles)
    
    def generate_temperature_profile(self, pressure, surface_temp, deep_temp):
        """Generate realistic temperature profile"""
        thermocline_depth = np.random.uniform(100, 500)
        thermocline_thickness = np.random.uniform(50, 200)
        
        temperature = np.zeros_like(pressure)
        for i, p in enumerate(pressure):
            if p < thermocline_depth:
                temperature[i] = surface_temp
            elif p < thermocline_depth + thermocline_thickness:
                # Linear decrease through thermocline
                frac = (p - thermocline_depth) / thermocline_thickness
                temperature[i] = surface_temp - frac * (surface_temp - deep_temp)
            else:
                temperature[i] = deep_temp
                
        # Add some noise
        temperature += np.random.normal(0, 0.1, len(temperature))
        return temperature
    
    def generate_salinity_profile(self, pressure):
        """Generate realistic salinity profile"""
        surface_salinity = np.random.uniform(33, 37)
        deep_salinity = np.random.uniform(34.5, 35)
        
        salinity = np.ones_like(pressure) * surface_salinity
        # Slight increase with depth
        salinity += (pressure / 2000) * (deep_salinity - surface_salinity)
        salinity += np.random.normal(0, 0.01, len(pressure))
        
        return salinity

    def render_sidebar(self):
        """Render the sidebar with navigation and filters"""
        with st.sidebar:
            st.markdown('<div class="main-header">🌊 ARGO AI</div>', unsafe_allow_html=True)
            
            # Navigation
            st.markdown("### Navigation")
            page = st.radio("", ["Home", "Data Explorer", "Fleet Monitoring", "AI Chat", "Visualization", "Upload Data"])
            
            # Data Filters
            st.markdown("---")
            st.markdown("### Data Filters")
            
            # Dataset selection
            dataset = st.selectbox("Dataset", ["All Data", "Sample Data", "Uploaded Data", "ARGO Core", "ARGO BGC"])
            
            # Region selection
            region = st.selectbox("Region", [
                "Global", "Pacific Ocean", "Atlantic Ocean", "Indian Ocean", 
                "Southern Ocean", "Arctic Ocean", "Custom Region"
            ])
            
            if region == "Custom Region":
                col1, col2 = st.columns(2)
                with col1:
                    lat_min, lat_max = st.slider("Latitude Range", -90.0, 90.0, (-90.0, 90.0))
                with col2:
                    lon_min, lon_max = st.slider("Longitude Range", -180.0, 180.0, (-180.0, 180.0))
            
            # Parameter selection
            st.markdown("### Parameters")
            parameters = st.multiselect("Select Parameters", 
                                      ["Temperature", "Salinity", "Oxygen", "Chlorophyll", "Nitrate"],
                                      default=["Temperature", "Salinity"])
            
            # Quality filter
            quality_range = st.slider("Data Quality", 0, 100, (80, 100))
            
            # Time range
            st.markdown("### Time Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("End Date", datetime.now())
            
            return page, {
                'dataset': dataset,
                'region': region,
                'parameters': parameters,
                'quality_range': quality_range,
                'start_date': start_date,
                'end_date': end_date
            }
    
    def render_home_page(self, filters):
        """Render the home page with overview"""
        st.markdown('<div class="main-header">🌊 ARGO AI Dashboard</div>', unsafe_allow_html=True)
        st.markdown("### All-Forward Audio Ocean Data Discovery & Visualization")
        
        # File upload section at the top of home page
        self.render_file_upload_section()
        
        # Get combined data
        floats_combined, profiles_combined = self.get_combined_data()
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_floats = len(floats_combined)
            st.metric("Total Floats", f"{total_floats:,}")
        
        with col2:
            total_profiles = len(profiles_combined)
            st.metric("Profiles Collected", f"{total_profiles:,}")
        
        with col3:
            active_floats = len(floats_combined[floats_combined['status'] == 'Active'])
            st.metric("Active Floats", active_floats)
        
        with col4:
            if len(profiles_combined) > 0:
                avg_temp = profiles_combined['temperature'].apply(lambda x: x[0] if len(x) > 0 else 0).mean()
                st.metric("Avg Surface Temp", f"{avg_temp:.1f}°C")
            else:
                st.metric("Avg Surface Temp", "N/A")
        
        with col5:
            uploaded_count = len(st.session_state.uploaded_files)
            st.metric("Uploaded Files", uploaded_count)
        
        # Ocean map
        st.markdown("---")
        st.markdown("### ARGO Float Locations")
        self.render_argo_map(floats_combined)
        
        # Recent activity
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Recent Data Activity")
            self.render_recent_activity(profiles_combined)
        
        with col2:
            st.markdown("### Quick Actions")
            if st.button("🚀 Explore Data", use_container_width=True):
                st.session_state.current_page = "Data Explorer"
                st.rerun()
            
            if st.button("🤖 AI Assistant", use_container_width=True):
                st.session_state.current_page = "AI Chat"
                st.rerun()
            
            if st.button("📊 Visualize", use_container_width=True):
                st.session_state.current_page = "Visualization"
                st.rerun()
            
            if st.button("📁 Upload More", use_container_width=True):
                st.session_state.current_page = "Upload Data"
                st.rerun()
    
    def render_file_upload_section(self):
        """Render file upload section on home page"""
        st.markdown("---")
        st.markdown('<div class="section-header">📁 Upload ARGO Data Files</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload ARGO data files (NetCDF, .nc, .nc4, BGC files)",
                type=['nc', 'nc4', 'bgc', 'gz', 'zip'],
                accept_multiple_files=True,
                key="home_upload"
            )
        
        with col2:
            if st.button("Process Uploaded Files", type="primary"):
                if uploaded_files:
                    self.process_uploaded_files(uploaded_files)
                else:
                    st.warning("Please select files to upload")
    
    def process_uploaded_files(self, uploaded_files):
        """Process all uploaded files"""
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Process based on file type
                        if uploaded_file.name.endswith(('.nc', '.nc4')):
                            self.process_netcdf_file(tmp_path, uploaded_file.name)
                        elif uploaded_file.name.endswith('.bgc'):
                            self.process_bgc_file(tmp_path, uploaded_file.name)
                        elif uploaded_file.name.endswith('.gz'):
                            self.process_gzip_file(tmp_path, uploaded_file.name)
                        elif uploaded_file.name.endswith('.zip'):
                            self.process_zip_file(tmp_path, uploaded_file.name)
                        
                        # Add to processed files list
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"Processed: {uploaded_file.name}")
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def process_netcdf_file(self, file_path, filename):
        """Process NetCDF file and extract data"""
        try:
            with xr.open_dataset(file_path) as ds:
                # Extract float information
                float_id = int(filename.split('_')[0]) if '_' in filename else len(st.session_state.uploaded_floats) + 3000000
                
                # Determine ocean based on coordinates
                lat = float(ds.LATITUDE.values) if 'LATITUDE' in ds.variables else np.random.uniform(-90, 90)
                lon = float(ds.LONGITUDE.values) if 'LONGITUDE' in ds.variables else np.random.uniform(-180, 180)
                ocean = self.get_ocean_from_coords(lat, lon)
                
                new_float = {
                    'float_id': float_id,
                    'ocean': ocean,
                    'latitude': lat,
                    'longitude': lon,
                    'status': 'Active',
                    'last_transmission': datetime.now(),
                    'profiles_count': 1,
                    'data_quality': 85.0,
                    'data_source': 'Uploaded'
                }
                
                # Add to uploaded floats
                if st.session_state.uploaded_floats.empty:
                    st.session_state.uploaded_floats = pd.DataFrame([new_float])
                else:
                    st.session_state.uploaded_floats = pd.concat([
                        st.session_state.uploaded_floats, 
                        pd.DataFrame([new_float])
                    ], ignore_index=True)
                
                # Extract profile data
                if 'TEMP' in ds.variables and 'PRES' in ds.variables:
                    pressure = ds.PRES.values.flatten()
                    temperature = ds.TEMP.values.flatten()
                    salinity = ds.PSAL.values.flatten() if 'PSAL' in ds.variables else self.generate_salinity_profile(pressure)
                    
                    new_profile = {
                        'profile_id': len(st.session_state.uploaded_profiles) + 100000,
                        'float_id': float_id,
                        'cycle_number': 1,
                        'date': datetime.now(),
                        'latitude': lat,
                        'longitude': lon,
                        'pressure_levels': pressure,
                        'temperature': temperature,
                        'salinity': salinity,
                        'quality_score': 85.0,
                        'data_source': 'Uploaded'
                    }
                    
                    if st.session_state.uploaded_profiles.empty:
                        st.session_state.uploaded_profiles = pd.DataFrame([new_profile])
                    else:
                        st.session_state.uploaded_profiles = pd.concat([
                            st.session_state.uploaded_profiles, 
                            pd.DataFrame([new_profile])
                        ], ignore_index=True)
                        
        except Exception as e:
            st.error(f"Error processing NetCDF file {filename}: {str(e)}")
    
    def process_bgc_file(self, file_path, filename):
        """Process BGC file (simplified)"""
        # Similar to NetCDF processing but for BGC specific parameters
        self.process_netcdf_file(file_path, filename)
    
    def process_gzip_file(self, file_path, filename):
        """Process gzipped file"""
        try:
            with gzip.open(file_path, 'rb') as f:
                content = f.read()
                # Save decompressed content and process
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                    tmp_file.write(content)
                    self.process_netcdf_file(tmp_file.name, filename.replace('.gz', ''))
                    os.unlink(tmp_file.name)
        except Exception as e:
            st.error(f"Error processing gzip file {filename}: {str(e)}")
    
    def process_zip_file(self, file_path, filename):
        """Process zip file containing multiple data files"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                extract_dir = tempfile.mkdtemp()
                zip_ref.extractall(extract_dir)
                
                # Process all extracted files
                for extracted_file in Path(extract_dir).rglob('*'):
                    if extracted_file.is_file():
                        if extracted_file.suffix in ['.nc', '.nc4', '.bgc']:
                            self.process_netcdf_file(str(extracted_file), extracted_file.name)
                
                # Clean up
                import shutil
                shutil.rmtree(extract_dir)
        except Exception as e:
            st.error(f"Error processing zip file {filename}: {str(e)}")
    
    def get_ocean_from_coords(self, lat, lon):
        """Determine ocean from latitude and longitude"""
        if -90 <= lat <= -60:  # Southern Ocean
            return 'Southern'
        elif 60 <= lat <= 90:  # Arctic Ocean
            return 'Arctic'
        elif -60 <= lat <= 60:
            if -180 <= lon <= -70 or 20 <= lon <= 180:  # Pacific Ocean
                return 'Pacific'
            elif -70 <= lon <= 20:  # Atlantic Ocean
                return 'Atlantic'
            elif 20 <= lon <= 120:  # Indian Ocean
                return 'Indian'
        return 'Unknown'
    
    def render_argo_map(self, floats_data):
        """Render ARGO float locations on a map"""
        if floats_data.empty:
            st.info("No float data available")
            return
            
        # Create a Folium map with OpenStreetMap tiles to avoid CORS issues
        m = folium.Map(
            location=[0, 0], 
            zoom_start=2,
            tiles='OpenStreetMap'
        )
        
        # Color coding by ocean and data source
        ocean_colors = {
            'Pacific': 'blue',
            'Atlantic': 'green',
            'Indian': 'orange',
            'Southern': 'red',
            'Arctic': 'purple',
            'Unknown': 'gray'
        }
        
        for _, float_data in floats_data.iterrows():
            color = ocean_colors.get(float_data['ocean'], 'gray')
            # Different marker for uploaded data
            icon_color = 'red' if float_data.get('data_source', 'Sample') == 'Uploaded' else color
            
            folium.CircleMarker(
                location=[float_data['latitude'], float_data['longitude']],
                radius=8,
                popup=f"""
                Float ID: {float_data['float_id']}<br>
                Ocean: {float_data['ocean']}<br>
                Status: {float_data['status']}<br>
                Profiles: {float_data['profiles_count']}<br>
                Source: {float_data.get('data_source', 'Sample')}
                """,
                color=icon_color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                tooltip=f"Float {float_data['float_id']}"
            ).add_to(m)
        
        # Display the map
        folium_static(m, width=900, height=500)
    
    def render_recent_activity(self, profiles_data):
        """Render recent data activity"""
        if profiles_data.empty:
            st.info("No profile data available")
            return
            
        # Get recent profiles (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_profiles = profiles_data[
            profiles_data['date'] >= recent_cutoff
        ]
        
        if len(recent_profiles) > 0:
            # Group by date and data source
            recent_profiles['date_only'] = recent_profiles['date'].dt.date
            daily_counts = recent_profiles.groupby(['date_only', 'data_source']).size().reset_index(name='count')
            
            fig = px.line(daily_counts, x='date_only', y='count', color='data_source',
                         title='Profiles Collected by Source (Last 30 Days)',
                         labels={'date_only': 'Date', 'count': 'Profiles Count'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent activity data available")
    
    def render_data_explorer(self, filters):
        """Render the data explorer page"""
        st.markdown("### 📊 Data Explorer")
        
        # Get combined data
        floats_combined, profiles_combined = self.get_combined_data()
        
        if floats_combined.empty:
            st.info("No data available. Please upload ARGO data files.")
            return
            
        # Filter data based on selections
        filtered_floats = floats_combined.copy()
        filtered_profiles = profiles_combined.copy()
        
        # Apply dataset filter
        if filters['dataset'] == 'Uploaded Data':
            filtered_floats = filtered_floats[filtered_floats.get('data_source', 'Sample') == 'Uploaded']
        elif filters['dataset'] == 'Sample Data':
            filtered_floats = filtered_floats[filtered_floats.get('data_source', 'Sample') == 'Sample']
        
        # Apply region filter
        if filters['region'] != 'Global':
            if filters['region'] == 'Custom Region':
                # Apply custom lat/lon filters
                pass  # Implementation needed
            else:
                ocean_map = {
                    'Pacific Ocean': 'Pacific',
                    'Atlantic Ocean': 'Atlantic',
                    'Indian Ocean': 'Indian',
                    'Southern Ocean': 'Southern',
                    'Arctic Ocean': 'Arctic'
                }
                target_ocean = ocean_map.get(filters['region'])
                if target_ocean:
                    filtered_floats = filtered_floats[filtered_floats['ocean'] == target_ocean]
        
        # Apply quality filter
        filtered_floats = filtered_floats[
            filtered_floats['data_quality'].between(filters['quality_range'][0], filters['quality_range'][1])
        ]
        
        filtered_profiles = filtered_profiles[
            filtered_profiles['float_id'].isin(filtered_floats['float_id'])
        ]
        
        # Display filtered results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Floats", len(filtered_floats))
        
        with col2:
            st.metric("Filtered Profiles", len(filtered_profiles))
        
        with col3:
            avg_quality = filtered_floats['data_quality'].mean()
            st.metric("Average Quality", f"{avg_quality:.1f}%")
        
        with col4:
            sample_count = len(filtered_floats[filtered_floats.get('data_source', 'Sample') == 'Sample'])
            uploaded_count = len(filtered_floats[filtered_floats.get('data_source', 'Sample') == 'Uploaded'])
            st.metric("Data Sources", f"S: {sample_count}, U: {uploaded_count}")
        
        # Data table
        st.markdown("### Float Data Table")
        display_columns = ['float_id', 'ocean', 'latitude', 'longitude', 'status', 'profiles_count', 'data_quality', 'data_source']
        st.dataframe(filtered_floats[display_columns], use_container_width=True)
        
        # Profile visualization
        st.markdown("### Profile Visualization")
        if len(filtered_profiles) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_float = st.selectbox("Select Float", filtered_floats['float_id'].unique())
            
            with col2:
                float_profiles = filtered_profiles[filtered_profiles['float_id'] == selected_float]
                if len(float_profiles) > 0:
                    selected_profile = st.selectbox("Select Profile", float_profiles['profile_id'].unique())
                else:
                    st.info("No profiles available for selected float")
                    return
            
            profile_data = float_profiles[float_profiles['profile_id'] == selected_profile].iloc[0]
            
            # Create profile plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=profile_data['temperature'], 
                    y=profile_data['pressure_levels'],
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title='Temperature Profile',
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Pressure (dbar)',
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=profile_data['salinity'], 
                    y=profile_data['pressure_levels'],
                    mode='lines',
                    name='Salinity',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title='Salinity Profile',
                    xaxis_title='Salinity (PSU)',
                    yaxis_title='Pressure (dbar)',
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No profile data available for the selected filters")
    
    def render_fleet_monitoring(self, filters):
        """Render fleet monitoring page"""
        st.markdown("### 🚢 Fleet Monitoring")
        
        # Get combined data
        floats_combined, profiles_combined = self.get_combined_data()
        
        if floats_combined.empty:
            st.info("No data available. Please upload ARGO data files.")
            return
        
        # Fleet overview by ocean and data source
        ocean_stats = floats_combined.groupby(['ocean', 'data_source']).agg({
            'float_id': 'count',
            'data_quality': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(ocean_stats, x='ocean', y='float_id', color='data_source',
                        title='Floats by Ocean and Data Source', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(floats_combined, names='ocean', 
                        title='Fleet Distribution by Ocean')
            st.plotly_chart(fig, use_container_width=True)
        
        # Status monitoring
        st.markdown("### Float Status Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_counts = floats_combined['status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title='Float Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Data quality by ocean
            fig = px.box(floats_combined, x='ocean', y='data_quality',
                        title='Data Quality Distribution by Ocean')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Active floats timeline
            recent_profiles = profiles_combined[
                profiles_combined['date'] >= (datetime.now() - timedelta(days=90))
            ]
            if len(recent_profiles) > 0:
                monthly_activity = recent_profiles.groupby([
                    recent_profiles['date'].dt.to_period('M'),
                    'data_source'
                ]).size().reset_index(name='count')
                monthly_activity['date'] = monthly_activity['date'].astype(str)
                
                fig = px.line(monthly_activity, x='date', y='count', color='data_source',
                             title='Monthly Profile Activity')
                st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_chat(self, filters):
        """Render AI chat interface"""
        st.markdown("### 🤖 AI Ocean Data Assistant")
        st.markdown("Hello! I'm your AI assistant for ARGO ocean data discovery. I can help you find temperature profiles, analyze ocean trends, explore specific fields, and much more.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your AI assistant for ARGO ocean data discovery. How can I help you today?"}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know about the ocean data?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                response = self.generate_ai_response(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Example queries
        st.markdown("### Try asking:")
        example_queries = [
            "Show me temperature profiles in the Pacific Ocean",
            "Find salinity variations at different depths",
            "Compare data from uploaded files vs sample data",
            "What's the average surface temperature?",
            "Show me the most active floats"
        ]
        
        cols = st.columns(3)
        for i, query in enumerate(example_queries):
            with cols[i % 3]:
                if st.button(query, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)
                    with st.chat_message("assistant"):
                        response = self.generate_ai_response(query)
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    def generate_ai_response(self, prompt):
        """Generate AI response based on user prompt and actual data"""
        prompt_lower = prompt.lower()
        floats_combined, profiles_combined = self.get_combined_data()
        
        if floats_combined.empty:
            return "I don't have any ARGO data to analyze yet. Please upload some ARGO data files to get started!"
        
        # Simple rule-based responses with actual data analysis
        if "temperature" in prompt_lower and "pacific" in prompt_lower:
            pacific_floats = floats_combined[floats_combined['ocean'] == 'Pacific']
            pacific_profiles = profiles_combined[
                profiles_combined['float_id'].isin(pacific_floats['float_id'])
            ]
            
            if len(pacific_profiles) > 0:
                surface_temps = []
                for temp_profile in pacific_profiles['temperature']:
                    if len(temp_profile) > 0:
                        surface_temps.append(temp_profile[0])
                
                if surface_temps:
                    avg_surface_temp = np.mean(surface_temps)
                    return f"The average surface temperature in the Pacific Ocean is {avg_surface_temp:.1f}°C. I found {len(pacific_profiles)} profiles from {len(pacific_floats)} floats in this region."
        
        elif "salinity" in prompt_lower:
            all_salinity = []
            for sal_profile in profiles_combined['salinity']:
                if len(sal_profile) > 0:
                    all_salinity.extend(sal_profile)
            
            if len(all_salinity) > 0:
                avg_salinity = np.mean(all_salinity)
                return f"The average salinity across all profiles is {avg_salinity:.2f} PSU. This is based on {len(profiles_combined)} profiles from {len(floats_combined)} floats."
        
        elif "active" in prompt_lower or "status" in prompt_lower:
            active_count = len(floats_combined[floats_combined['status'] == 'Active'])
            inactive_count = len(floats_combined[floats_combined['status'] == 'Inactive'])
            return f"There are {active_count} active floats and {inactive_count} inactive floats in the dataset."
        
        elif "uploaded" in prompt_lower and "sample" in prompt_lower:
            uploaded_count = len(floats_combined[floats_combined.get('data_source', 'Sample') == 'Uploaded'])
            sample_count = len(floats_combined[floats_combined.get('data_source', 'Sample') == 'Sample'])
            return f"The dataset contains {sample_count} sample floats and {uploaded_count} floats from uploaded files."
        
        elif "quality" in prompt_lower:
            avg_quality = floats_combined['data_quality'].mean()
            best_ocean = floats_combined.groupby('ocean')['data_quality'].mean().idxmax()
            return f"The average data quality is {avg_quality:.1f}%. The {best_ocean} Ocean has the highest quality data on average."
        
        # Default response
        return f"I've analyzed your query about ARGO ocean data. I currently have data from {len(floats_combined)} floats across {len(floats_combined['ocean'].unique())} oceans, with {len(profiles_combined)} total profiles. What specific aspect would you like to explore further?"
    
    def render_visualization(self, filters):
        """Render visualization page"""
        st.markdown("### 📊 Data Visualization")
        
        # Get combined data
        floats_combined, profiles_combined = self.get_combined_data()
        
        if floats_combined.empty:
            st.info("No data available. Please upload ARGO data files.")
            return
        
        # Visualization options
        viz_type = st.selectbox("Select Visualization Type", [
            "Ocean Distribution Map",
            "Temperature Heatmap",
            "Salinity Distribution",
            "Profile Comparisons",
            "Time Series Analysis"
        ])
        
        if viz_type == "Ocean Distribution Map":
            self.render_ocean_distribution_map(floats_combined)
        
        elif viz_type == "Temperature Heatmap":
            self.render_temperature_heatmap(profiles_combined)
        
        elif viz_type == "Salinity Distribution":
            self.render_salinity_distribution(profiles_combined)
        
        elif viz_type == "Profile Comparisons":
            self.render_profile_comparisons(profiles_combined)
        
        elif viz_type == "Time Series Analysis":
            self.render_time_series_analysis(profiles_combined)
    
    def render_ocean_distribution_map(self, floats_data):
        """Render ocean distribution map"""
        st.markdown("### Ocean Distribution of ARGO Floats")
        
        # Create a more detailed map with OpenStreetMap to avoid CORS
        m = folium.Map(location=[0, 0], zoom_start=2, tiles='OpenStreetMap')
        
        # Add ocean boundaries
        ocean_boundaries = {
            'Pacific': {'lat_range': (-60, 60), 'lon_range': (120, -60)},
            'Atlantic': {'lat_range': (-60, 60), 'lon_range': (-80, 20)},
            'Indian': {'lat_range': (-60, 30), 'lon_range': (20, 120)},
            'Southern': {'lat_range': (-90, -60), 'lon_range': (-180, 180)},
            'Arctic': {'lat_range': (60, 90), 'lon_range': (-180, 180)}
        }
        
        for ocean, bounds in ocean_boundaries.items():
            folium.Rectangle(
                bounds=[[bounds['lat_range'][0], bounds['lon_range'][0]], 
                        [bounds['lat_range'][1], bounds['lon_range'][1]]],
                color='blue',
                fill=True,
                fillOpacity=0.1,
                popup=ocean,
                tooltip=ocean
            ).add_to(m)
        
        # Add float markers
        for _, float_data in floats_data.iterrows():
            folium.CircleMarker(
                location=[float_data['latitude'], float_data['longitude']],
                radius=6,
                popup=f"Float {float_data['float_id']} - {float_data['ocean']}",
                color='red',
                fill=True,
                fillOpacity=0.7,
                tooltip=f"Float {float_data['float_id']}"
            ).add_to(m)
        
        folium_static(m, width=900, height=500)
    
    def render_temperature_heatmap(self, profiles_data):
        """Render temperature heatmap using Plotly instead of Mapbox"""
        st.markdown("### Temperature Distribution")
        
        if len(profiles_data) > 0:
            # Extract surface temperatures
            surface_temps = []
            latitudes = []
            longitudes = []
            
            for _, profile in profiles_data.iterrows():
                if len(profile['temperature']) > 0:
                    surface_temps.append(profile['temperature'][0])
                    latitudes.append(profile['latitude'])
                    longitudes.append(profile['longitude'])
            
            if len(surface_temps) > 0:
                # Create a scatter plot instead of heatmap to avoid Mapbox issues
                fig = px.scatter_geo(
                    lat=latitudes,
                    lon=longitudes,
                    color=surface_temps,
                    size=[5] * len(surface_temps),
                    title="Surface Temperature Distribution",
                    color_continuous_scale="Viridis",
                    labels={'color': 'Temperature (°C)'}
                )
                fig.update_geos(projection_type="equirectangular")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_salinity_distribution(self, profiles_data):
        """Render salinity distribution"""
        st.markdown("### Salinity Distribution by Ocean")
        
        if len(profiles_data) > 0:
            # Get ocean information from float data
            floats_combined, _ = self.get_combined_data()
            profiles_with_ocean = profiles_data.merge(
                floats_combined[['float_id', 'ocean']], on='float_id', how='left'
            )
            
            # Extract surface salinity
            surface_salinity = []
            oceans = []
            
            for _, profile in profiles_with_ocean.iterrows():
                if len(profile['salinity']) > 0:
                    surface_salinity.append(profile['salinity'][0])
                    oceans.append(profile['ocean'])
            
            if len(surface_salinity) > 0:
                fig = px.box(x=oceans, y=surface_salinity, 
                            title="Surface Salinity Distribution by Ocean")
                fig.update_layout(xaxis_title="Ocean", yaxis_title="Salinity (PSU)")
                st.plotly_chart(fig, use_container_width=True)
    
    def render_profile_comparisons(self, profiles_data):
        """Render profile comparisons"""
        st.markdown("### Profile Comparisons")
        
        if len(profiles_data) > 0:
            # Select profiles to compare
            profile_ids = st.multiselect("Select profiles to compare", 
                                       profiles_data['profile_id'].unique(),
                                       max_selections=3)
            
            if len(profile_ids) > 0:
                selected_profiles = profiles_data[profiles_data['profile_id'].isin(profile_ids)]
                
                fig = go.Figure()
                
                for _, profile in selected_profiles.iterrows():
                    fig.add_trace(go.Scatter(
                        x=profile['temperature'],
                        y=profile['pressure_levels'],
                        mode='lines',
                        name=f"Profile {profile['profile_id']}",
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title='Temperature Profile Comparison',
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Pressure (dbar)',
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_time_series_analysis(self, profiles_data):
        """Render time series analysis"""
        st.markdown("### Time Series Analysis")
        
        if len(profiles_data) > 0:
            # Convert dates if needed
            if not pd.api.types.is_datetime64_any_dtype(profiles_data['date']):
                profiles_data['date'] = pd.to_datetime(profiles_data['date'])
            
            # Extract surface temperature over time
            dates = []
            surface_temps = []
            
            for _, profile in profiles_data.iterrows():
                if len(profile['temperature']) > 0:
                    dates.append(profile['date'])
                    surface_temps.append(profile['temperature'][0])
            
            if len(dates) > 0:
                time_series_df = pd.DataFrame({
                    'date': dates,
                    'surface_temperature': surface_temps
                })
                
                # Monthly averages
                monthly_avg = time_series_df.set_index('date').resample('M').mean().reset_index()
                
                fig = px.line(monthly_avg, x='date', y='surface_temperature',
                             title='Monthly Average Surface Temperature')
                st.plotly_chart(fig, use_container_width=True)
    
    def render_upload_data(self, filters):
        """Render dedicated upload data page"""
        st.markdown("### 📁 Upload ARGO Data Files")
        st.markdown("Upload your ARGO data files in NetCDF (.nc, .nc4), BGC, or compressed (.gz, .zip) formats.")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Choose ARGO data files",
            type=['nc', 'nc4', 'bgc', 'gz', 'zip'],
            accept_multiple_files=True,
            help="Upload one or more ARGO data files"
        )
        
        # File processing options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                if uploaded_files:
                    self.process_uploaded_files(uploaded_files)
                else:
                    st.warning("Please select files to upload")
        
        with col2:
            if st.button("🔄 Clear Uploaded Data", use_container_width=True):
                st.session_state.uploaded_floats = pd.DataFrame()
                st.session_state.uploaded_profiles = pd.DataFrame()
                st.session_state.uploaded_files = []
                st.success("Uploaded data cleared!")
        
        with col3:
            if st.button("📥 Download Sample", use_container_width=True):
                self.download_sample_data()
        
        # Display uploaded files
        st.markdown("### Uploaded Files")
        if st.session_state.uploaded_files:
            for filename in st.session_state.uploaded_files:
                st.success(f"✓ {filename}")
        else:
            st.info("No files uploaded yet")
        
        # Data preview
        if not st.session_state.uploaded_floats.empty:
            st.markdown("### Uploaded Data Preview")
            st.dataframe(st.session_state.uploaded_floats, use_container_width=True)
    
    def download_sample_data(self):
        """Provide sample data download"""
        # Create a simple sample NetCDF file for download
        sample_data = """
        # Sample ARGO Data Format
        # This is a simplified representation of ARGO NetCDF data
        # Actual files would contain binary NetCDF data
        
        FLOAT_ID: 2900001
        LATITUDE: 35.5
        LONGITUDE: -120.5
        JULD: 2023-01-15
        TEMP: [20.1, 19.8, 18.5, 17.2, 16.0, ...]
        PSAL: [33.5, 33.6, 33.7, 33.8, 33.9, ...]
        PRES: [0, 10, 20, 30, 40, ...]
        """
        
        st.download_button(
            label="📥 Download Sample Data Format",
            data=sample_data,
            file_name="argo_sample_format.txt",
            mime="text/plain"
        )
    
    def run(self):
        """Main application runner"""
        # Initialize session state for current page
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"
        
        # Render sidebar and get filters
        page, filters = self.render_sidebar()
        
        # Update current page
        st.session_state.current_page = page
        
        # Render the selected page
        if page == "Home":
            self.render_home_page(filters)
        elif page == "Data Explorer":
            self.render_data_explorer(filters)
        elif page == "Fleet Monitoring":
            self.render_fleet_monitoring(filters)
        elif page == "AI Chat":
            self.render_ai_chat(filters)
        elif page == "Visualization":
            self.render_visualization(filters)
        elif page == "Upload Data":
            self.render_upload_data(filters)

# Run the application
if __name__ == "__main__":
    dashboard = ArgoAIDashboard()
    dashboard.run()