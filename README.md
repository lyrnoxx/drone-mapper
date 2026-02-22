# DJI Drone Mapper

Real-time orthomosaic map generation from DJI drone imagery with pose graph optimization and multi-band blending.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

## Features

- **Real-time Map Stitching** — Tile-based multi-band Laplacian pyramid blending for seamless orthomosaic generation
- **Pose Graph Optimization** — SIFT feature matching with sparse least-squares solver to correct GPS drift
- **A/B Comparison Mode** — Side-by-side view of raw GPS vs optimized maps with quantitative metrics (Seam SSIM, correction magnitude)
- **Flight Path Overlay** — Visualize drone trajectory on the generated map
- **Interactive Dashboard** — Per-frame correction charts, flight path comparison, displacement analysis, and performance profiling
- **Upload & Simulate** — Drag-and-drop image upload or run simulation from a local image folder
- **Configurable Parameters** — Adjustable map resolution, blend bands, tile size, and optimizer weights (GPS/match weighting, IoU threshold)

## Architecture

```
dji_mapper_app.py          # Streamlit UI (sidebar controls, map display, dashboards)
backend/
├── core.py                # Engine: PoseExtractor, PoseGraphOptimizer, MultiBandMap2D
├── mapper_service.py      # Service layer: singleton, processing pipeline, metrics
├── api.py                 # FastAPI endpoints (WebSocket support)
└── __init__.py
```

### Core Components

| Component | Description |
|-----------|-------------|
| `PoseExtractor` | Parses DJI XMP metadata (GPS, gimbal angles) → 4×4 pose matrix in NED frame via UTM projection |
| `PoseGraphOptimizer` | SIFT + FLANN matching between overlapping frames, sparse LSQR block adjustment |
| `MultiBandMap2D` | Tile-based mapper with Laplacian pyramid blending, flight path tracking, memory-efficient chunked rendering |

## Quick Start

### Prerequisites

- Python 3.10+
- DJI drone images with GPS EXIF/XMP metadata

### Installation

```bash
git clone https://github.com/lyrnoxx/drone-mapper.git
cd dji-mapper
pip install -r requirements.txt
```

### Run

```bash
streamlit run dji_mapper_app.py
```

Open `http://localhost:8501` in your browser.

### Usage

1. **Upload** — Drag and drop DJI JPEG images using the sidebar uploader
2. **Simulate** — Place images in an `images-true/` folder and click "Run Simulation"
3. **Choose Mode** — Select Raw GPS, Pose Graph Optimization, or A/B Comparison
4. **Explore** — Pan/zoom the interactive map, review metrics and performance dashboards


## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Map Resolution | 0.5 m/px | Spatial resolution of the output map |
| Blend Bands | 2 | Laplacian pyramid levels for multi-band blending |
| Tile Size | 512 px | Size of internal map tile chunks |
| GPS Weight | 1.0 | Trust in raw GPS positions during optimization |
| Match Weight | 2.0 | Trust in SIFT feature-match constraints |
| Min Matches | 15 | Minimum SIFT matches to accept a frame pair |
| IoU Threshold | 0.10 | Minimum overlap ratio to consider frames as neighbors |

## Tech Stack

- **Frontend:** Streamlit, Plotly
- **Computer Vision:** OpenCV (SIFT, homography), NumPy
- **Optimization:** SciPy (sparse LSQR)
- **Geospatial:** PyProj (UTM projection)
- **API:** FastAPI + WebSocket (optional)

## License

MIT

---

*2026 — Jugal Alan*
