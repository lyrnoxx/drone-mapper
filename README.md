# Drone Mapper Engine

A real-time drone mapping system built with Python, FastAPI, and Streamlit. The application processes drone imagery to generate georeferenced maps, utilizing satellite-guided alignment to correct GPS drift and orientation errors.

## Core Features
- **Real-time Map Generation**: Automated stitching of drone imagery into a unified orthomosaic.
- **Satellite-Guided Refinement**: Integrated alignment with ESRI satellite imagery to improve positional accuracy (GPS & Yaw).
- **Interactive Visualization**: High-resolution map exploration using Plotly-based interactive charts.
- **Automated Processing Pipeline**: Support for batch image processing and live simulation modes.

## Technical Specifications
- **Backend**: Python 3.x, FastAPI.
- **Frontend**: Streamlit.
- **Image Processing**: OpenCV, NumPy, SciPy.
- **Geospatial Processing**: PyProj for coordinate transformations.
- **Visualization**: Plotly, PIL.

## Installation and Usage

### 1. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution
Start the Streamlit application:
```bash
streamlit run dji_mapper_app.py
```

---
*Developed by Jugal Alan*
