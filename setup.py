#!/usr/bin/env python3
# setup.py

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="argo-ai-system",
    version="1.0.0",
    description="Advanced ARGO float data analysis and visualization system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/argo-ai-system",
    author="ARGO AI Team",
    author_email="team@argo-ai.org",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="argo, oceanography, data-analysis, visualization, nlp",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <3.12",
    install_requires=[
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "numpy>=1.21.0",
        "xarray>=2023.1.0",
        "netCDF4>=1.6.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "scikit-learn>=1.2.0",
        "folium>=0.14.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.3.0",
        ],
        "production": [
            "gunicorn>=21.0.0",
            "psutil>=5.9.0",
            "uvicorn>=0.22.0",
        ],
        "geospatial": [
            "geopandas>=0.13.0",
            "shapely>=2.0.0",
            "pyproj>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "argo-ai=src.main:main",
            "argo-db-init=scripts.initialize_database:main",
        ],
    },
    package_data={
        "argo_ai": [
            "config/*.yaml",
            "data/*.csv",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/argo-ai-system/issues",
        "Source": "https://github.com/your-org/argo-ai-system",
    },
)