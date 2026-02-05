import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
import io
import sys
import os

# Ensure backend can be imported
sys.path.append(os.getcwd())

from backend.mapper_service import MapperService

st.set_page_config(page_title="DJI Mapper", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for UI Tweaks
st.markdown("""
    <style>
        /* Hide Deploy Button */
        .stDeployButton {
            display: none;
        }
        /* Reduce Top Padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        /* Reduce Header Height */
        header {
            height: 2.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Use cache_resource to maintain the singleton across reruns
@st.cache_resource
def get_service():
    return MapperService.get_instance()

service = get_service()

st.title("DJI Mapper Engine")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    with st.expander("Camera Parameters", expanded=False):
        focal_len = st.number_input("Focal Length (mm)", value=24.0, step=0.1)
        sensor_w = st.number_input("Sensor Width (mm)", value=36.0, step=0.1)
        img_w = st.number_input("Image Width (px)", value=4000, step=100)
        
        if st.button("Apply Params"):
            new_fx = service.update_parameters(focal_len, sensor_w, img_w)
            st.success(f"Updated fx={new_fx:.1f}")

    uploaded_files = st.file_uploader(
        "Upload Drone Images", 
        type=["jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    if st.button("Reset Map", type="primary"):
        service.reset_map()
        st.session_state.processed_files = set()
        st.success("Map reset!")
        st.rerun()

    st.divider()
    st.header("Advanced Features")
    enable_sat = st.checkbox("Satellite-Guided Alignment", value=service.mapper.enable_refinement, help="Uses ESRI satellite imagery to correct GPS drift and yaw errors.")
    service.mapper.enable_refinement = enable_sat

    st.divider()
    st.header("Simulation")
    sim_speed = st.slider("Speed (sec)", 0.1, 2.0, 0.5)
    
    if st.button("Run Simulation \u25b6\ufe0f"):
        st.session_state.is_simulating = True

import plotly.express as px

# Main Area
# --- MAIN LAYOUT ---
# Section 1: Map (Dominant)
map_placeholder = st.empty()

def update_map_display():
    map_bytes = service.get_map_image()
    if map_bytes:
        # Load image
        image = Image.open(io.BytesIO(map_bytes))
        img_array = np.array(image)
        
        # Create interactive figure
        fig = px.imshow(img_array)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            dragmode="pan", # Enable panning by default
            height=600 # Fixed height for consistency
        )
        
        # Display
        map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_{time.time()}")
    else:
        map_placeholder.info("No map data yet. Upload images or Run Simulation to start.")

# Initial map render
update_map_display()

st.divider()

# Section 2: Logs and Info (Bottom)
b_col1, b_col2 = st.columns([1, 1])

with b_col1:
    with st.expander("System Status", expanded=True):
        st.markdown(f"""
        **Operational Status**
        - **Engine:** \u2705 Online
        - **Drone Connection:** \u26A0\uFE0F Disconnected (Simulation Mode)
        - **Map Resolution:** 0.5 m/px
        - **Active Tile Chunks:** {len(service.tiles) if hasattr(service, 'tiles') else 0}
        """)
        if st.session_state.get("is_simulating"):
            st.info("\u25B6\uFE0F Simulation Running...")
        else:
            st.success("System Ready for Input")

with b_col2:
    st.subheader("Processing Log")
    # Use expander to save space
    with st.expander("Show Details", expanded=True):
        log_container = st.container()
    
    # SIMULATION LOGIC
    if st.session_state.get("is_simulating"):
        sim_folder = "images-true"
        if os.path.exists(sim_folder):
            images = sorted([f for f in os.listdir(sim_folder) if f.lower().endswith(('.jpg', '.jpeg'))])
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, img_name in enumerate(images):
                img_path = os.path.join(sim_folder, img_name)
                status_text.text(f"Simulating: {img_name}")
                
                with open(img_path, "rb") as f:
                    bytes_data = f.read()
                    
                result = service.process_image(bytes_data)
                
                if result["status"] == "success":
                    # Only show success logs if we want, or keep it minimal
                    log_container.success(f"Processed: {img_name}")
                else:
                    log_container.warning(f"Ignored: {img_name} ({result.get('message')})")
                
                # LIVE UPDATE: Refresh map in col2
                update_map_display()
                
                time.sleep(sim_speed)
                progress_bar.progress((i + 1) / len(images))
            
            st.session_state.is_simulating = False
            status_text.text("Simulation Complete")
            st.rerun()
        else:
            st.error(f"Folder '{sim_folder}' not found!")
            st.session_state.is_simulating = False

    # UPLOAD LOGIC
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process new files
        # In a real app, we might want to track which files are already processed
        # For this simple version, we'll process what is in the uploader
        # Note: Streamlit re-uploads on interaction, so we need a way to not re-process.
        # Simple fix: Use session state to track processed files.
        
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
            
        for i, file in enumerate(uploaded_files):
            if file.name not in st.session_state.processed_files:
                status_text.text(f"Processing {file.name}...")
                
                # Read file
                bytes_data = file.read()
                
                # Process
                result = service.process_image(bytes_data)
                
                if result["status"] == "success":
                    log_container.success(f"Processed: {file.name}")
                    st.session_state.processed_files.add(file.name)
                elif result["status"] == "ignored":
                    log_container.warning(f"Ignored: {file.name} ({result['message']})")
                    # Also mark as processed so we don't try again
                    st.session_state.processed_files.add(file.name)
                else:
                    log_container.error(f"Error {file.name}: {result.get('message')}")
                
                # Update map after each upload processing
                update_map_display()
                
                progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Ready")

st.divider()
st.caption("2026 - Jugal Alan - Powered by Streamlit")
