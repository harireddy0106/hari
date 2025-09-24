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
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: 2rem;
    }
    .ai-message {
        background-color: #f0f0f0;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class ArgoAIDashboard:
    def __init__(self):
        self.setup_logging()
        self.initialize_data()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_data(self):
        """Initialize sample data for demonstration"""
        # Sample ARGO float data
        self.floats_data = self.generate_sample_floats()
        self.profiles_data = self.generate_sample_profiles()
        
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
                    'data_quality': np.random.uniform(70, 95)
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
                    'quality_score': np.random.uniform(80, 95)
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
            st.markdown('<div class="main-header">🌊 ARGO AI Dashboard</div>', unsafe_allow_html=True)
            
            # Navigation
            st.markdown("### Navigation")
            page = st.radio("", ["Home", "Data Explorer", "Fleet Monitoring", "AI Chat", "Visualization", "Upload Data"])
            
            # Data Filters
            st.markdown("---")
            st.markdown("### Data Filters")
            
            # Dataset selection
            dataset = st.selectbox("Dataset", ["ARGO Core", "ARGO BGC", "All Data"])
            
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
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_floats = len(self.floats_data)
            st.metric("Total Floats", f"{total_floats:,}")
        
        with col2:
            total_profiles = len(self.profiles_data)
            st.metric("Profiles Collected", f"{total_profiles:,}")
        
        with col3:
            active_floats = len(self.floats_data[self.floats_data['status'] == 'Active'])
            st.metric("Active Floats", active_floats)
        
        with col4:
            avg_temp = self.profiles_data['temperature'].apply(lambda x: x[0] if len(x) > 0 else 0).mean()
            st.metric("Avg Surface Temp", f"{avg_temp:.1f}°C")
        
        # Ocean map
        st.markdown("---")
        st.markdown("### ARGO Float Locations")
        self.render_argo_map()
        
        # Recent activity
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Recent Data Activity")
            self.render_recent_activity()
        
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
    
    def render_argo_map(self):
        """Render ARGO float locations on a map"""
        # Create a Folium map
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        # Color coding by ocean
        ocean_colors = {
            'Pacific': 'blue',
            'Atlantic': 'green',
            'Indian': 'orange',
            'Southern': 'red',
            'Arctic': 'purple'
        }
        
        for _, float_data in self.floats_data.iterrows():
            color = ocean_colors.get(float_data['ocean'], 'gray')
            
            folium.CircleMarker(
                location=[float_data['latitude'], float_data['longitude']],
                radius=8,
                popup=f"""
                Float ID: {float_data['float_id']}<br>
                Ocean: {float_data['ocean']}<br>
                Status: {float_data['status']}<br>
                Profiles: {float_data['profiles_count']}<br>
                Quality: {float_data['data_quality']:.1f}%
                """,
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
        
        # Display the map
        folium_static(m, width=900, height=500)
    
    def render_recent_activity(self):
        """Render recent data activity"""
        # Get recent profiles (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_profiles = self.profiles_data[
            self.profiles_data['date'] >= recent_cutoff
        ]
        
        if len(recent_profiles) > 0:
            # Group by date
            daily_counts = recent_profiles.groupby(
                recent_profiles['date'].dt.date
            ).size().reset_index(name='count')
            
            fig = px.line(daily_counts, x='date', y='count', 
                         title='Profiles Collected (Last 30 Days)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent activity data available")
    
    def render_data_explorer(self, filters):
        """Render the data explorer page"""
        st.markdown("### 📊 Data Explorer")
        
        # Filter data based on selections
        filtered_floats = self.floats_data.copy()
        filtered_profiles = self.profiles_data.copy()
        
        # Apply region filter
        if filters['region'] != 'Global':
            if filters['region'] == 'Custom Region':
                # Apply custom lat/lon filters (would need to be implemented)
                pass
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
                    filtered_profiles = filtered_profiles[
                        filtered_profiles['float_id'].isin(filtered_floats['float_id'])
                    ]
        
        # Apply quality filter
        filtered_floats = filtered_floats[
            filtered_floats['data_quality'].between(filters['quality_range'][0], filters['quality_range'][1])
        ]
        
        # Display filtered results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filtered Floats", len(filtered_floats))
        
        with col2:
            st.metric("Filtered Profiles", len(filtered_profiles))
        
        with col3:
            avg_quality = filtered_floats['data_quality'].mean()
            st.metric("Average Quality", f"{avg_quality:.1f}%")
        
        # Data table
        st.markdown("### Float Data Table")
        display_columns = ['float_id', 'ocean', 'latitude', 'longitude', 'status', 'profiles_count', 'data_quality']
        st.dataframe(filtered_floats[display_columns], use_container_width=True)
        
        # Profile visualization
        st.markdown("### Profile Visualization")
        if len(filtered_profiles) > 0:
            selected_float = st.selectbox("Select Float", filtered_floats['float_id'].unique())
            float_profiles = filtered_profiles[filtered_profiles['float_id'] == selected_float]
            
            if len(float_profiles) > 0:
                selected_profile = st.selectbox("Select Profile", 
                                              float_profiles['profile_id'].unique())
                
                profile_data = float_profiles[float_profiles['profile_id'] == selected_profile].iloc[0]
                
                # Create profile plots
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=profile_data['temperature'], 
                        y=profile_data['pressure_levels'],
                        mode='lines',
                        name='Temperature'
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
                        name='Salinity'
                    ))
                    fig.update_layout(
                        title='Salinity Profile',
                        xaxis_title='Salinity (PSU)',
                        yaxis_title='Pressure (dbar)',
                        yaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_fleet_monitoring(self, filters):
        """Render fleet monitoring page"""
        st.markdown("### 🚢 Fleet Monitoring")
        
        # Fleet overview by ocean
        ocean_stats = self.floats_data.groupby('ocean').agg({
            'float_id': 'count',
            'data_quality': 'mean',
            'profiles_count': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(ocean_stats, x='ocean', y='float_id',
                        title='Floats by Ocean')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(ocean_stats, values='float_id', names='ocean',
                        title='Fleet Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Status monitoring
        st.markdown("### Float Status")
        status_counts = self.floats_data['status'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title='Float Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recent activity timeline
            recent_profiles = self.profiles_data[
                self.profiles_data['date'] >= (datetime.now() - timedelta(days=30))
            ]
            daily_activity = recent_profiles.groupby(recent_profiles['date'].dt.date).size()
            
            fig = px.line(x=daily_activity.index, y=daily_activity.values,
                         title='Recent Profile Activity')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Quality distribution
            fig = px.histogram(self.floats_data, x='data_quality',
                             title='Data Quality Distribution')
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
            "Compare chlorophyll levels between different oceans",
            "What's the average surface temperature in the Indian Ocean?",
            "Show me the most active floats in the Atlantic"
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
        """Generate AI response based on user prompt"""
        prompt_lower = prompt.lower()
        
        # Simple rule-based responses (in a real app, this would use an AI model)
        if "temperature" in prompt_lower and "pacific" in prompt_lower:
            pacific_floats = self.floats_data[self.floats_data['ocean'] == 'Pacific']
            pacific_profiles = self.profiles_data[
                self.profiles_data['float_id'].isin(pacific_floats['float_id'])
            ]
            
            if len(pacific_profiles) > 0:
                avg_surface_temp = np.mean([temp[0] for temp in pacific_profiles['temperature'] if len(temp) > 0])
                return f"The average surface temperature in the Pacific Ocean is {avg_surface_temp:.1f}°C. I found {len(pacific_profiles)} profiles from {len(pacific_floats)} floats in this region."
        
        elif "salinity" in prompt_lower:
            all_salinity = []
            for sal_profile in self.profiles_data['salinity']:
                if len(sal_profile) > 0:
                    all_salinity.extend(sal_profile)
            
            if len(all_salinity) > 0:
                avg_salinity = np.mean(all_salinity)
                return f"The average salinity across all profiles is {avg_salinity:.2f} PSU. Salinity typically ranges from 33-37 PSU in most ocean regions."
        
        elif "compare" in prompt_lower and "chlorophyll" in prompt_lower:
            return "Chlorophyll data shows interesting patterns across different ocean basins. The highest concentrations are typically found in upwelling regions and coastal areas where nutrient availability supports phytoplankton growth."
        
        elif "active" in prompt_lower and "atlantic" in prompt_lower:
            atlantic_floats = self.floats_data[
                (self.floats_data['ocean'] == 'Atlantic') & 
                (self.floats_data['status'] == 'Active')
            ]
            return f"There are {len(atlantic_floats)} active floats in the Atlantic Ocean, collecting valuable data on ocean conditions."
        
        else:
            return "I can help you explore ARGO ocean data! Try asking about specific parameters like temperature or salinity, or inquire about data from particular ocean regions. I can also help you visualize trends and patterns in the data."
    
    def render_visualization(self, filters):
        """Render advanced visualization page"""
        st.markdown("### 📊 Advanced Visualization")
        
        # Visualization type selection
        viz_type = st.selectbox("Visualization Type", [
            "Ocean Map", "Profile Comparison", "Time Series", "Parameter Correlation"
        ])
        
        if viz_type == "Ocean Map":
            self.render_ocean_parameter_map()
        
        elif viz_type == "Profile Comparison":
            self.render_profile_comparison()
        
        elif viz_type == "Time Series":
            self.render_time_series()
        
        elif viz_type == "Parameter Correlation":
            self.render_parameter_correlation()
    
    def render_ocean_parameter_map(self):
        """Render parameter distribution on ocean map"""
        st.markdown("### Ocean Parameter Distribution")
        
        parameter = st.selectbox("Select Parameter", ["Temperature", "Salinity"])
        depth_level = st.slider("Depth Level (dbar)", 0, 2000, 0, 100)
        
        # Create a map with parameter values
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        for _, profile in self.profiles_data.iterrows():
            if len(profile['pressure_levels']) > 0:
                # Find closest depth level
                depth_idx = np.argmin(np.abs(profile['pressure_levels'] - depth_level))
                
                if parameter == "Temperature" and len(profile['temperature']) > depth_idx:
                    value = profile['temperature'][depth_idx]
                    color = self.temperature_to_color(value)
                elif parameter == "Salinity" and len(profile['salinity']) > depth_idx:
                    value = profile['salinity'][depth_idx]
                    color = self.salinity_to_color(value)
                else:
                    continue
                
                folium.CircleMarker(
                    location=[profile['latitude'], profile['longitude']],
                    radius=6,
                    popup=f"{parameter}: {value:.2f}",
                    color=color,
                    fill=True,
                    fillColor=color
                ).add_to(m)
        
        folium_static(m, width=900, height=500)
    
    def temperature_to_color(self, temp):
        """Convert temperature to color (blue to red scale)"""
        # Normalize temperature to 0-1 range (-2°C to 30°C)
        normalized = (temp + 2) / 32
        normalized = max(0, min(1, normalized))
        
        # Blue to red color scale
        if normalized < 0.5:
            # Blue to green
            r = 0
            g = int(255 * normalized * 2)
            b = 255 - int(255 * normalized * 2)
        else:
            # Green to red
            r = int(255 * (normalized - 0.5) * 2)
            g = 255 - int(255 * (normalized - 0.5) * 2)
            b = 0
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def salinity_to_color(self, salinity):
        """Convert salinity to color (light to dark blue)"""
        # Normalize salinity to 0-1 range (33-37 PSU)
        normalized = (salinity - 33) / 4
        normalized = max(0, min(1, normalized))
        
        # Light blue to dark blue
        r = 0
        g = int(100 + 155 * (1 - normalized))
        b = int(200 + 55 * normalized)
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def render_profile_comparison(self):
        """Render profile comparison visualization"""
        st.markdown("### Profile Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            float1 = st.selectbox("Float 1", self.floats_data['float_id'].unique())
            profiles1 = self.profiles_data[self.profiles_data['float_id'] == float1]
            profile1 = st.selectbox("Profile 1", profiles1['profile_id'].unique())
        
        with col2:
            float2 = st.selectbox("Float 2", self.floats_data['float_id'].unique())
            profiles2 = self.profiles_data[self.profiles_data['float_id'] == float2]
            profile2 = st.selectbox("Profile 2", profiles2['profile_id'].unique())
        
        # Get profile data
        data1 = profiles1[profiles1['profile_id'] == profile1].iloc[0]
        data2 = profiles2[profiles2['profile_id'] == profile2].iloc[0]
        
        # Create comparison plot
        fig = go.Figure()
        
        # Temperature profiles
        fig.add_trace(go.Scatter(
            x=data1['temperature'], y=data1['pressure_levels'],
            mode='lines', name=f'Float {float1} Temp',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=data2['temperature'], y=data2['pressure_levels'],
            mode='lines', name=f'Float {float2} Temp',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Temperature Profile Comparison',
            xaxis_title='Temperature (°C)',
            yaxis_title='Pressure (dbar)',
            yaxis=dict(autorange='reversed')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_time_series(self):
        """Render time series visualization"""
        st.markdown("### Time Series Analysis")
        
        selected_float = st.selectbox("Select Float", self.floats_data['float_id'].unique())
        float_profiles = self.profiles_data[self.profiles_data['float_id'] == selected_float]
        
        if len(float_profiles) > 0:
            # Extract surface values over time
            dates = []
            surface_temps = []
            surface_salinity = []
            
            for _, profile in float_profiles.iterrows():
                if len(profile['temperature']) > 0 and len(profile['salinity']) > 0:
                    dates.append(profile['date'])
                    surface_temps.append(profile['temperature'][0])
                    surface_salinity.append(profile['salinity'][0])
            
            if len(dates) > 0:
                # Create time series plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates, y=surface_temps,
                    mode='lines+markers', name='Surface Temperature',
                    yaxis='y1'
                ))
                
                fig.add_trace(go.Scatter(
                    x=dates, y=surface_salinity,
                    mode='lines+markers', name='Surface Salinity',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title=f'Time Series - Float {selected_float}',
                    xaxis_title='Date',
                    yaxis=dict(title='Temperature (°C)', titlefont=dict(color='red')),
                    yaxis2=dict(
                        title='Salinity (PSU)', titlefont=dict(color='blue'),
                        overlaying='y', side='right'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_parameter_correlation(self):
        """Render parameter correlation visualization"""
        st.markdown("### Parameter Correlation")
        
        # Extract surface values from all profiles
        surface_temps = []
        surface_salinity = []
        
        for _, profile in self.profiles_data.iterrows():
            if len(profile['temperature']) > 0 and len(profile['salinity']) > 0:
                surface_temps.append(profile['temperature'][0])
                surface_salinity.append(profile['salinity'][0])
        
        if len(surface_temps) > 0:
            # Create scatter plot
            fig = px.scatter(
                x=surface_temps, y=surface_salinity,
                title='Temperature vs Salinity Correlation',
                labels={'x': 'Surface Temperature (°C)', 'y': 'Surface Salinity (PSU)'}
            )
            
            # Add trendline
            z = np.polyfit(surface_temps, surface_salinity, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=np.sort(surface_temps), 
                y=p(np.sort(surface_temps)),
                mode='lines',
                name='Trendline',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            correlation = np.corrcoef(surface_temps, surface_salinity)[0,1]
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    def render_upload_data(self, filters):
        """Render data upload page"""
        st.markdown("### 📁 Upload Ocean Data")
        
        st.info("Upload NetCDF files containing ARGO float data for analysis")
        
        uploaded_file = st.file_uploader("Choose a NetCDF file", type=['nc', 'nc4'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Try to open the NetCDF file
                with xr.open_dataset(tmp_path) as ds:
                    st.success(f"Successfully loaded NetCDF file: {uploaded_file.name}")
                    
                    # Display file information
                    st.markdown("#### File Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Dimensions:**")
                        for dim, size in ds.dims.items():
                            st.write(f"- {dim}: {size}")
                    
                    with col2:
                        st.write("**Variables:**")
                        for var in ds.variables:
                            st.write(f"- {var}")
                    
                    # Show preview of data
                    st.markdown("#### Data Preview")
                    st.dataframe(ds.to_dataframe().head(), use_container_width=True)
                    
                    # Process button
                    if st.button("Process Data", type="primary"):
                        with st.spinner("Processing data..."):
                            # Simulate data processing
                            self.process_uploaded_data(ds)
                            st.success("Data processed successfully! It's now available for analysis.")
            
            except Exception as e:
                st.error(f"Error reading NetCDF file: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
    
    def process_uploaded_data(self, dataset):
        """Process uploaded NetCDF data (simulated)"""
        # In a real implementation, this would extract data from the NetCDF file
        # and add it to the application's data structures
        self.logger.info("Processing uploaded dataset")
        
        # Simulate adding new data
        new_float_id = self.floats_data['float_id'].max() + 1
        new_float = {
            'float_id': new_float_id,
            'ocean': 'Pacific',  # Default
            'latitude': 0,
            'longitude': 0,
            'status': 'Active',
            'last_transmission': datetime.now(),
            'profiles_count': 1,
            'data_quality': 85.0
        }
        
        self.floats_data = pd.concat([self.floats_data, pd.DataFrame([new_float])], ignore_index=True)
        
        # Add sample profile
        new_profile_id = self.profiles_data['profile_id'].max() + 1
        pressure_levels = np.linspace(0, 2000, 50)
        temperature = self.generate_temperature_profile(pressure_levels, 25, 2)
        salinity = self.generate_salinity_profile(pressure_levels)
        
        new_profile = {
            'profile_id': new_profile_id,
            'float_id': new_float_id,
            'cycle_number': 1,
            'date': datetime.now(),
            'latitude': 0,
            'longitude': 0,
            'pressure_levels': pressure_levels,
            'temperature': temperature,
            'salinity': salinity,
            'quality_score': 85.0
        }
        
        self.profiles_data = pd.concat([self.profiles_data, pd.DataFrame([new_profile])], ignore_index=True)
    
    def run(self):
        """Main application runner"""
        # Initialize session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"
        
        # Render sidebar and get filters
        page, filters = self.render_sidebar()
        
        # Update current page
        st.session_state.current_page = page
        
        # Render the selected page
        if st.session_state.current_page == "Home":
            self.render_home_page(filters)
        elif st.session_state.current_page == "Data Explorer":
            self.render_data_explorer(filters)
        elif st.session_state.current_page == "Fleet Monitoring":
            self.render_fleet_monitoring(filters)
        elif st.session_state.current_page == "AI Chat":
            self.render_ai_chat(filters)
        elif st.session_state.current_page == "Visualization":
            self.render_visualization(filters)
        elif st.session_state.current_page == "Upload Data":
            self.render_upload_data(filters)

# Run the application
if __name__ == "__main__":
    dashboard = ArgoAIDashboard()
    dashboard.run()