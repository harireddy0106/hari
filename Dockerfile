# Dockerfile
# ARGO AI System Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgeos-dev \
    proj-bin \
    libproj-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data backups config

# Create non-root user
RUN groupadd -r argo && useradd -r -g argo argo
RUN chown -R argo:argo /app
USER argo

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')"

# Default command (Streamlit)
CMD ["streamlit", "run", "streamlit_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]