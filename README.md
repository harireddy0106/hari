# 🌊 FloatChat - AI-Powered ARGO Ocean Data Explorer

FloatChat is an intelligent conversational interface for exploring ARGO float oceanographic data using natural language queries.

## Features

- **Natural Language Interface**: Chat with ocean data using simple English
- **AI-Powered Queries**: LLM converts questions to SQL automatically
- **Interactive Visualizations**: Maps, profiles, time series
- **Multi-format Export**: CSV, JSON, NetCDF
- **Real-time Data**: Live ARGO data feeds
- **Indian Ocean Focus**: INCOIS specialized

## Quick Start

### Installation

```bash
# Extract the zip file
unzip floatchat-argo-ai.zip
cd floatchat-argo-ai

# Run setup
chmod +x scripts/setup.sh
./scripts/setup.sh

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the app
source venv/bin/activate
streamlit run streamlit_app/main.py
```
