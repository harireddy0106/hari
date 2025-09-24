# visualization/plot_generator.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AdvancedArgoPlotGenerator:
    def __init__(self):
        self.color_scales = {
            'temperature': 'thermal',
            'salinity': 'haline', 
            'oxygen': 'oxy',
            'chlorophyll': 'algae',
            'nitrate': 'matter',
            'quality': 'viridis'
        }
        
        self.region_boundaries = {
            'indian_ocean': {'lat': [0, 30], 'lon': [40, 120]},
            'arabian_sea': {'lat': [10, 25], 'lon': [50, 75]},
            'bay_of_bengal': {'lat': [10, 25], 'lon': [80, 100]},
            'global': {'lat': [-90, 90], 'lon': [-180, 180]}
        }
    
    def create_interactive_map(self, data: List[Dict], map_type: str = 'marker') -> folium.Map:
        """Create advanced interactive maps with multiple visualization options"""
        if not data:
            return self._create_empty_map()
        
        df = pd.DataFrame(data)
        
        # Calculate map center and bounds
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=4,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Add different map types
        if map_type == 'heatmap' and len(df) > 10:
            self._add_heatmap_layer(m, df)
        elif map_type == 'cluster':
            self._add_cluster_layer(m, df)
        else:
            self._add_enhanced_markers(m, df)
        
        # Add layer control
        self._add_map_controls(m)
        
        return m
    
    def _add_enhanced_markers(self, map_obj: folium.Map, df: pd.DataFrame):
        """Add enhanced markers with popups and coloring"""
        # Color markers by quality score if available
        if 'quality_score' in df.columns:
            df['quality_color'] = df['quality_score'].apply(self._get_quality_color)
        else:
            df['quality_color'] = 'blue'
        
        for _, row in df.iterrows():
            # Create enhanced popup
            popup_content = self._create_enhanced_popup(row)
            
            # Create custom icon
            icon = folium.Icon(
                icon='tint',
                icon_color='white',
                color=row['quality_color'],
                prefix='fa'
            )
            
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=400),
                icon=icon,
                tooltip=f"Float {row.get('float_id', 'Unknown')}",
                draggable=False
            ).add_to(map_obj)
    
    def _add_heatmap_layer(self, map_obj: folium.Map, df: pd.DataFrame):
        """Add heatmap layer for density visualization"""
        heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, gradient={0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}).add_to(map_obj)
    
    def _add_cluster_layer(self, map_obj: folium.Map, df: pd.DataFrame):
        """Add marker cluster layer"""
        marker_cluster = MarkerCluster().add_to(map_obj)
        
        for _, row in df.iterrows():
            popup_content = self._create_basic_popup(row)
            icon = folium.Icon(icon='tint', color='blue', prefix='fa')
            
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=icon
            ).add_to(marker_cluster)
    
    def _create_enhanced_popup(self, row) -> str:
        """Create enhanced popup with comprehensive information"""
        popup_parts = [
            f"<b>Float ID:</b> {row.get('float_id', 'Unknown')}",
            f"<b>Date:</b> {row.get('profile_date', 'Unknown')}",
            f"<b>Location:</b> {row.get('latitude', 0):.2f}°N, {row.get('longitude', 0):.2f}°E",
            f"<b>Cycle:</b> {row.get('cycle_number', 'N/A')}"
        ]
        
        # Add quality information
        if 'quality_score' in row and row['quality_score'] is not None:
            quality_status = self._get_quality_status(row['quality_score'])
            popup_parts.append(f"<b>Quality:</b> {quality_status} ({row['quality_score']:.1f}%)")
        
        # Add parameter availability
        available_params = []
        for param in ['temperature', 'salinity', 'oxygen', 'chlorophyll']:
            if f'{param}_values' in row and row[f'{param}_values']:
                available_params.append(param)
        
        if available_params:
            popup_parts.append(f"<b>Parameters:</b> {', '.join(available_params)}")
        
        return "<br>".join(popup_parts)
    
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 90: return 'green'
        elif score >= 80: return 'blue'
        elif score >= 70: return 'orange'
        else: return 'red'
    
    def _get_quality_status(self, score: float) -> str:
        """Get quality status text"""
        if score >= 90: return 'Excellent'
        elif score >= 80: return 'Good'
        elif score >= 70: return 'Fair'
        elif score >= 60: return 'Poor'
        else: return 'Unusable'
    
    def create_comprehensive_profile_plot(self, profiles: List[Dict]) -> go.Figure:
        """Create comprehensive profile visualization with multiple parameters"""
        if not profiles:
            return self._create_empty_plot("No profile data available")
        
        # Create subplots for different parameters
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Profile', 'Salinity Profile', 'Oxygen Profile', 'Quality Indicators'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot each profile
        for i, profile in enumerate(profiles[:5]):  # Limit to 5 profiles for clarity
            self._add_profile_to_plot(fig, profile, i)
        
        fig.update_layout(
            title="Comprehensive Ocean Profiles",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def _add_profile_to_plot(self, fig: go.Figure, profile: Dict, profile_idx: int):
        """Add a single profile to the comprehensive plot"""
        pressure = profile.get('pressure_levels', [])
        if not pressure:
            return
        
        # Temperature subplot (row 1, col 1)
        if profile.get('temperature_values'):
            fig.add_trace(
                go.Scatter(
                    x=profile['temperature_values'],
                    y=pressure,
                    name=f"Temp Profile {profile_idx+1}",
                    line=dict(color=px.colors.qualitative.Set1[profile_idx % 10], width=2),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Salinity subplot (row 1, col 2)
        if profile.get('salinity_values'):
            fig.add_trace(
                go.Scatter(
                    x=profile['salinity_values'],
                    y=pressure,
                    name=f"Salinity Profile {profile_idx+1}",
                    line=dict(color=px.colors.qualitative.Set2[profile_idx % 10], width=2),
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # Oxygen subplot (row 2, col 1)
        if profile.get('oxygen_values'):
            fig.add_trace(
                go.Scatter(
                    x=profile['oxygen_values'],
                    y=pressure,
                    name=f"Oxygen Profile {profile_idx+1}",
                    line=dict(color=px.colors.qualitative.Set3[profile_idx % 10], width=2),
                    showlegend=True
                ),
                row=2, col=1
            )
    
    def create_temporal_analysis_plot(self, data: List[Dict], parameter: str = 'temperature') -> go.Figure:
        """Create advanced temporal analysis plot"""
        if not data:
            return self._create_empty_plot("No temporal data available")
        
        df = pd.DataFrame(data)
        
        # Extract time series data
        time_series_data = self._extract_temporal_data(df, parameter)
        
        if not time_series_data:
            return self._create_empty_plot(f"No {parameter} data available for temporal analysis")
        
        # Create figure with multiple visualization types
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{parameter.title()} Time Series',
                'Monthly Averages',
                'Seasonal Patterns',
                'Data Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Time series plot
        self._add_time_series_plot(fig, time_series_data, parameter, row=1, col=1)
        
        # Monthly averages
        self._add_monthly_averages(fig, time_series_data, parameter, row=1, col=2)
        
        # Seasonal patterns
        self._add_seasonal_analysis(fig, time_series_data, parameter, row=2, col=1)
        
        # Data distribution
        self._add_distribution_plot(fig, time_series_data, parameter, row=2, col=2)
        
        fig.update_layout(
            title=f"Temporal Analysis of {parameter.title()}",
            height=700,
            template="plotly_white"
        )
        
        return fig
    
    def _extract_temporal_data(self, df: pd.DataFrame, parameter: str) -> pd.DataFrame:
        """Extract and prepare temporal data"""
        temporal_data = []
        
        for _, row in df.iterrows():
            if pd.notna(row.get('profile_date')) and row.get(f'{parameter}_values'):
                values = row[f'{parameter}_values']
                if values and len(values) > 0:
                    # Use surface value (first measurement)
                    surface_value = values[0] if values else None
                    if surface_value is not None:
                        temporal_data.append({
                            'date': row['profile_date'],
                            'value': surface_value,
                            'float_id': row.get('float_id', 'Unknown'),
                            'latitude': row.get('latitude'),
                            'longitude': row.get('longitude')
                        })
        
        return pd.DataFrame(temporal_data) if temporal_data else pd.DataFrame()
    
    def _add_time_series_plot(self, fig: go.Figure, data: pd.DataFrame, parameter: str, row: int, col: int):
        """Add time series plot to figure"""
        if data.empty:
            return
        
        # Group by float for better visualization
        for float_id in data['float_id'].unique():
            float_data = data[data['float_id'] == float_id]
            fig.add_trace(
                go.Scatter(
                    x=float_data['date'],
                    y=float_data['value'],
                    name=f"Float {float_id}",
                    mode='lines+markers',
                    marker=dict(size=4)
                ),
                row=row, col=col
            )
    
    def create_quality_dashboard(self, data: List[Dict]) -> go.Figure:
        """Create comprehensive quality assessment dashboard"""
        if not data:
            return self._create_empty_plot("No quality data available")
        
        df = pd.DataFrame(data)
        
        # Create quality dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Quality Score Distribution',
                'Quality vs Time',
                'Spatial Quality Map',
                'Parameter Quality Comparison',
                'Data Mode Distribution',
                'Quality Trends'
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}, {"type": "scattergeo"}],
                   [{"type": "bar"}, {"type": "pie"}, {"type": "box"}]]
        )
        
        # Quality distribution
        if 'quality_score' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['quality_score'], nbinsx=20, name="Quality Scores"),
                row=1, col=1
            )
        
        # Quality vs Time
        if 'profile_date' in df.columns and 'quality_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['profile_date'],
                    y=df['quality_score'],
                    mode='markers',
                    name="Quality Over Time"
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="ARGO Data Quality Dashboard",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def _add_map_controls(self, map_obj: folium.Map):
        """Add layer controls to map"""
        folium.LayerControl().add_to(map_obj)
        
        # Add custom controls
        folium.LatLngPopup().add_to(map_obj)
    
    def _create_empty_map(self, center: Tuple[float, float] = (20, 80)) -> folium.Map:
        """Create empty map with informative message"""
        m = folium.Map(location=center, zoom_start=3)
        
        # Add informative message
        folium.Marker(
            center,
            icon=folium.DivIcon(html='<div style="color: red; font-size: 16px;">No data available</div>')
        ).add_to(m)
        
        return m
    
    def _create_empty_plot(self, message: str = "No data available") -> go.Figure:
        """Create empty plot with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            plot_bgcolor='white',
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    # Placeholder methods for additional features
    def _add_monthly_averages(self, fig: go.Figure, data: pd.DataFrame, parameter: str, row: int, col: int):
        """Add monthly averages plot to temporal analysis"""
        if data.empty:
            return
        
        # Extract month and year
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        
        # Calculate monthly averages
        monthly_avg = data.groupby('month')['value'].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_avg['month'],
                y=monthly_avg['value'],
                mode='lines+markers',
                name='Monthly Average',
                line=dict(color='red', width=3)
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Month", row=row, col=col)
        fig.update_yaxes(title_text=parameter.title(), row=row, col=col)

    def _add_seasonal_analysis(self, fig: go.Figure, data: pd.DataFrame, parameter: str, row: int, col: int):
        """Add seasonal analysis plot"""
        if data.empty:
            return
        
        # Define seasons
        def get_season(month):
            if month in [12, 1, 2]: return 'Winter'
            elif month in [3, 4, 5]: return 'Spring'
            elif month in [6, 7, 8]: return 'Summer'
            else: return 'Autumn'
        
        data['season'] = data['month'].apply(get_season)
        seasonal_avg = data.groupby('season')['value'].mean().reset_index()
        
        # Order seasons properly
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        seasonal_avg['season'] = pd.Categorical(seasonal_avg['season'], categories=season_order, ordered=True)
        seasonal_avg = seasonal_avg.sort_values('season')
        
        fig.add_trace(
            go.Bar(
                x=seasonal_avg['season'],
                y=seasonal_avg['value'],
                name='Seasonal Average',
                marker_color=['blue', 'green', 'red', 'orange']
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Season", row=row, col=col)
        fig.update_yaxes(title_text=f"Average {parameter}", row=row, col=col)

    def _add_distribution_plot(self, fig: go.Figure, data: pd.DataFrame, parameter: str, row: int, col: int):
        """Add distribution plot"""
        if data.empty:
            return
        
        fig.add_trace(
            go.Histogram(
                x=data['value'],
                nbinsx=30,
                name=f'{parameter} Distribution',
                marker_color='lightblue'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text=parameter.title(), row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    def create_vertical_section_plot(self, profiles: List[Dict], parameter: str = 'temperature') -> go.Figure:
        """Create vertical section plot showing parameter distribution along depth"""
        if not profiles:
            return self._create_empty_plot("No profile data available for vertical section")
        
        # Prepare data for vertical section
        depth_data = []
        for profile in profiles:
            if profile.get('pressure_levels') and profile.get(f'{parameter}_values'):
                pressures = profile['pressure_levels']
                values = profile[f'{parameter}_values']
                lat = profile.get('latitude', 0)
                lon = profile.get('longitude', 0)
                
                for pressure, value in zip(pressures, values):
                    if value is not None and not np.isnan(value):
                        depth_data.append({
                            'pressure': pressure,
                            'value': value,
                            'latitude': lat,
                            'longitude': lon,
                            'float_id': profile.get('float_id', 'Unknown')
                        })
        
        if not depth_data:
            return self._create_empty_plot(f"No {parameter} data available for vertical section")
        
        df = pd.DataFrame(depth_data)
        
        # Create contour plot for vertical section
        fig = go.Figure()
        
        # Sort by latitude for proper ordering
        df_sorted = df.sort_values('latitude')
        
        # Create contour plot
        fig.add_trace(
            go.Contour(
                z=df_sorted['value'],
                x=df_sorted['latitude'],
                y=df_sorted['pressure'],
                colorscale=self.color_scales.get(parameter, 'viridis'),
                colorbar=dict(title=parameter.title()),
                contours=dict(
                    coloring='heatmap',
                    showlines=True
                )
            )
        )
        
        fig.update_layout(
            title=f"Vertical Section - {parameter.title()}",
            xaxis_title="Latitude (°N)",
            yaxis_title="Pressure (dbar)",
            template="plotly_white",
            height=600
        )
        
        # Reverse y-axis to show depth increasing downward
        fig.update_yaxes(autorange="reversed")
        
        return fig

    def create_comparison_plot(self, profiles1: List[Dict], profiles2: List[Dict], 
                             label1: str = "Dataset 1", label2: str = "Dataset 2") -> go.Figure:
        """Create comparison plot between two datasets"""
        if not profiles1 and not profiles2:
            return self._create_empty_plot("No data available for comparison")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Temperature Comparison',
                'Salinity Comparison',
                'Oxygen Comparison',
                'Quality Score Comparison'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Compare different parameters
        self._add_parameter_comparison(fig, profiles1, profiles2, 'temperature', label1, label2, 1, 1)
        self._add_parameter_comparison(fig, profiles1, profiles2, 'salinity', label1, label2, 1, 2)
        self._add_parameter_comparison(fig, profiles1, profiles2, 'oxygen', label1, label2, 2, 1)
        self._add_quality_comparison(fig, profiles1, profiles2, label1, label2, 2, 2)
        
        fig.update_layout(
            title=f"Dataset Comparison: {label1} vs {label2}",
            height=700,
            template="plotly_white",
            showlegend=True
        )
        
        return fig

    def _add_parameter_comparison(self, fig: go.Figure, profiles1: List[Dict], profiles2: List[Dict], 
                                parameter: str, label1: str, label2: str, row: int, col: int):
        """Add parameter comparison to comparison plot"""
        # Extract surface values for comparison
        values1 = self._extract_surface_values(profiles1, parameter)
        values2 = self._extract_surface_values(profiles2, parameter)
        
        if values1:
            fig.add_trace(
                go.Box(y=values1, name=f"{label1} {parameter}", marker_color='blue', boxpoints='outliers'),
                row=row, col=col
            )
        
        if values2:
            fig.add_trace(
                go.Box(y=values2, name=f"{label2} {parameter}", marker_color='red', boxpoints='outliers'),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Dataset", row=row, col=col)
        fig.update_yaxes(title_text=parameter.title(), row=row, col=col)

    def _add_quality_comparison(self, fig: go.Figure, profiles1: List[Dict], profiles2: List[Dict], 
                              label1: str, label2: str, row: int, col: int):
        """Add quality score comparison"""
        quality1 = [p.get('quality_score', 0) for p in profiles1 if p.get('quality_score') is not None]
        quality2 = [p.get('quality_score', 0) for p in profiles2 if p.get('quality_score') is not None]
        
        if quality1:
            fig.add_trace(
                go.Histogram(x=quality1, name=label1, opacity=0.7, marker_color='blue', nbinsx=20),
                row=row, col=col
            )
        
        if quality2:
            fig.add_trace(
                go.Histogram(x=quality2, name=label2, opacity=0.7, marker_color='red', nbinsx=20),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Quality Score", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    def _extract_surface_values(self, profiles: List[Dict], parameter: str) -> List[float]:
        """Extract surface values from profiles"""
        values = []
        for profile in profiles:
            param_values = profile.get(f'{parameter}_values', [])
            if param_values and len(param_values) > 0:
                surface_value = param_values[0]
                if surface_value is not None and not np.isnan(surface_value):
                    values.append(surface_value)
        return values

    def create_region_analysis_plot(self, data: List[Dict], region: str = 'indian_ocean') -> go.Figure:
        """Create regional analysis plot for specific ocean region"""
        if not data:
            return self._create_empty_plot(f"No data available for {region} analysis")
        
        df = pd.DataFrame(data)
        
        # Filter data for the specified region
        region_bounds = self.region_boundaries.get(region, self.region_boundaries['global'])
        regional_data = df[
            (df['latitude'] >= region_bounds['lat'][0]) & 
            (df['latitude'] <= region_bounds['lat'][1]) &
            (df['longitude'] >= region_bounds['lon'][0]) & 
            (df['longitude'] <= region_bounds['lon'][1])
        ]
        
        if regional_data.empty:
            return self._create_empty_plot(f"No data available for {region} region")
        
        # Create comprehensive regional analysis
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                f'{region.replace("_", " ").title()} - Spatial Distribution',
                'Temperature Distribution',
                'Salinity Distribution',
                'Time Series Analysis',
                'Depth Profiles',
                'Parameter Correlations'
            ),
            specs=[[{"type": "scattergeo"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Spatial distribution
        self._add_regional_spatial_plot(fig, regional_data, region, 1, 1)
        
        # Parameter distributions
        self._add_parameter_distribution(fig, regional_data, 'temperature', 1, 2)
        self._add_parameter_distribution(fig, regional_data, 'salinity', 1, 3)
        
        # Time series
        self._add_regional_time_series(fig, regional_data, 2, 1)
        
        # Depth profiles
        self._add_regional_depth_profiles(fig, regional_data, 2, 2)
        
        # Correlations
        self._add_parameter_correlation(fig, regional_data, 2, 3)
        
        fig.update_layout(
            title=f"Regional Analysis - {region.replace('_', ' ').title()}",
            height=800,
            template="plotly_white"
        )
        
        return fig

    def _add_regional_spatial_plot(self, fig: go.Figure, data: pd.DataFrame, region: str, row: int, col: int):
        """Add regional spatial distribution plot"""
        fig.add_trace(
            go.Scattergeo(
                lon=data['longitude'],
                lat=data['latitude'],
                text=data['float_id'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data.get('quality_score', 80),
                    colorscale='Viridis',
                    colorbar=dict(title="Quality Score"),
                    showscale=True
                )
            ),
            row=row, col=col
        )

    def _add_parameter_distribution(self, fig: go.Figure, data: pd.DataFrame, parameter: str, row: int, col: int):
        """Add parameter distribution plot for regional analysis"""
        values = self._extract_surface_values(data.to_dict('records'), parameter)
        if values:
            fig.add_trace(
                go.Histogram(x=values, name=parameter.title(), nbinsx=30),
                row=row, col=col
            )
            fig.update_xaxes(title_text=parameter.title(), row=row, col=col)

    def _add_regional_time_series(self, fig: go.Figure, data: pd.DataFrame, row: int, col: int):
        """Add regional time series analysis"""
        if 'profile_date' in data.columns and 'temperature_values' in data.columns:
            # Extract surface temperature over time
            time_data = []
            for _, row in data.iterrows():
                if row.get('temperature_values') and len(row['temperature_values']) > 0:
                    time_data.append({
                        'date': row['profile_date'],
                        'temperature': row['temperature_values'][0]
                    })
            
            if time_data:
                time_df = pd.DataFrame(time_data)
                fig.add_trace(
                    go.Scatter(
                        x=time_df['date'],
                        y=time_df['temperature'],
                        mode='markers',
                        name='Surface Temperature'
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Date", row=row, col=col)
                fig.update_yaxes(title_text="Temperature (°C)", row=row, col=col)

    def _add_regional_depth_profiles(self, fig: go.Figure, data: pd.DataFrame, row: int, col: int):
        """Add regional depth profiles"""
        profiles = data.to_dict('records')
        for i, profile in enumerate(profiles[:3]):  # Show first 3 profiles
            if profile.get('pressure_levels') and profile.get('temperature_values'):
                fig.add_trace(
                    go.Scatter(
                        x=profile['temperature_values'],
                        y=profile['pressure_levels'],
                        name=f"Profile {i+1}",
                        mode='lines'
                    ),
                    row=row, col=col
                )
        fig.update_xaxes(title_text="Temperature (°C)", row=row, col=col)
        fig.update_yaxes(title_text="Pressure (dbar)", row=row, col=col, autorange="reversed")

    def _add_parameter_correlation(self, fig: go.Figure, data: pd.DataFrame, row: int, col: int):
        """Add parameter correlation plot"""
        # Extract surface temperature and salinity
        temp_values = self._extract_surface_values(data.to_dict('records'), 'temperature')
        sal_values = self._extract_surface_values(data.to_dict('records'), 'salinity')
        
        if temp_values and sal_values and len(temp_values) == len(sal_values):
            fig.add_trace(
                go.Scatter(
                    x=temp_values,
                    y=sal_values,
                    mode='markers',
                    name='T-S Diagram'
                ),
                row=row, col=col
            )
            fig.update_xaxes(title_text="Temperature (°C)", row=row, col=col)
            fig.update_yaxes(title_text="Salinity (PSU)", row=row, col=col)

    def create_animated_timeseries(self, data: List[Dict], parameter: str = 'temperature') -> go.Figure:
        """Create animated time series plot"""
        if not data:
            return self._create_empty_plot("No data available for animation")
        
        df = pd.DataFrame(data)
        
        # Ensure we have date information
        if 'profile_date' not in df.columns:
            return self._create_empty_plot("No date information available for animation")
        
        # Extract surface values
        animated_data = []
        for _, row in df.iterrows():
            if row.get(f'{parameter}_values') and len(row[f'{parameter}_values']) > 0:
                animated_data.append({
                    'date': row['profile_date'],
                    'value': row[f'{parameter}_values'][0],
                    'float_id': row.get('float_id', 'Unknown'),
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude')
                })
        
        if not animated_data:
            return self._create_empty_plot(f"No {parameter} data available for animation")
        
        anim_df = pd.DataFrame(animated_data)
        
        # Create animated plot
        fig = px.scatter(
            anim_df, 
            x='longitude', 
            y='latitude', 
            animation_frame=anim_df['date'].dt.strftime('%Y-%m'),
            color='value',
            size_max=10,
            color_continuous_scale=self.color_scales.get(parameter, 'viridis'),
            title=f"Animated {parameter.title()} Time Series"
        )
        
        fig.update_layout(
            geo=dict(
                projection_type='equirectangular',
                showland=True,
                landcolor='lightgreen',
                showocean=True,
                oceancolor='lightblue'
            ),
            height=600
        )
        
        return fig

    def export_plot(self, fig, filename: str, format: str = 'html', width: int = 1200, height: int = 800):
        """Export plot to various formats"""
        try:
            if format.lower() == 'html':
                fig.write_html(f"{filename}.html", config={'responsive': True})
            elif format.lower() == 'png':
                fig.write_image(f"{filename}.png", width=width, height=height)
            elif format.lower() == 'pdf':
                fig.write_image(f"{filename}.pdf", width=width, height=height)
            elif format.lower() == 'json':
                fig.write_json(f"{filename}.json")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Plot exported successfully: {filename}.{format}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting plot: {e}")
            return False

    def create_dashboard(self, data: List[Dict]) -> go.Figure:
        """Create comprehensive dashboard with multiple visualizations"""
        if not data:
            return self._create_empty_plot("No data available for dashboard")
        
        # Create a comprehensive dashboard with 3x3 grid
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Spatial Distribution', 'Temperature Time Series', 'Salinity Time Series',
                'Quality Distribution', 'Depth Profiles', 'Parameter Correlation',
                'Monthly Averages', 'Regional Analysis', 'Data Statistics'
            ),
            specs=[[{"type": "scattergeo"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        # Add various visualizations to the dashboard
        self._add_dashboard_visualizations(fig, data)
        
        fig.update_layout(
            title="ARGO Data Comprehensive Dashboard",
            height=1000,
            template="plotly_white",
            showlegend=True
        )
        
        return fig

    def _add_dashboard_visualizations(self, fig: go.Figure, data: List[Dict]):
        """Add various visualizations to the dashboard"""
        df = pd.DataFrame(data)
        
        # Spatial distribution (1,1)
        if not df.empty:
            fig.add_trace(
                go.Scattergeo(
                    lon=df['longitude'],
                    lat=df['latitude'],
                    mode='markers',
                    marker=dict(size=6, color=df.get('quality_score', 80), colorscale='Viridis')
                ),
                row=1, col=1
            )
        
        # Add more visualizations based on available data
        # ... (implementation would continue based on specific dashboard requirements)

# Utility functions for plot generation
def create_summary_report(plots: Dict[str, go.Figure], data_stats: Dict[str, Any]) -> str:
    """Create HTML summary report with embedded plots"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ARGO Data Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .plot-container { margin: 20px 0; border: 1px solid #ddd; padding: 10px; }
            .stats-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .stats-table th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>ARGO Data Analysis Report</h1>
        <p>Generated on: {generation_date}</p>
        
        <h2>Data Statistics</h2>
        {stats_table}
        
        <h2>Visualizations</h2>
    """.format(
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stats_table=_create_stats_table(data_stats)
    )
    
    # Add plots to HTML
    for plot_name, plot_fig in plots.items():
        plot_html = plot_fig.to_html(include_plotlyjs='cdn', div_id=plot_name)
        html_content += f"""
        <div class="plot-container">
            <h3>{plot_name}</h3>
            {plot_html}
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def _create_stats_table(stats: Dict[str, Any]) -> str:
    """Create HTML table from statistics dictionary"""
    if not stats:
        return "<p>No statistics available</p>"
    
    table_rows = ""
    for key, value in stats.items():
        table_rows += f"<tr><td>{key}</td><td>{value}</td></tr>"
    
    return f"""
    <table class="stats-table">
        <tr><th>Metric</th><th>Value</th></tr>
        {table_rows}
    </table>
    """