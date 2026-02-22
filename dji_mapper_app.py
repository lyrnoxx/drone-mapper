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

st.set_page_config(page_title="Drone Mapper", layout="wide", initial_sidebar_state="expanded")

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
        /* Move sidebar to the right by reversing flex layout */
        [data-testid="stAppViewContainer"] {
            flex-direction: row-reverse !important;
        }
        /* Move the expand button (appears when sidebar collapsed) to the right */
        button[data-testid="stExpandSidebarButton"] {
            position: fixed !important;
            right: 0.5rem !important;
            left: unset !important;
            top: 0.4rem !important;
            z-index: 999999 !important;
        }
        /* Override sidebar slide animation: slide from right instead of left */
        [data-testid="stSidebar"] {
            transition: margin-right 300ms, visibility 300ms !important;
            transition-property: margin-right, visibility !important;
        }
        /* Cap Processing Log height */
        .processing-log-container {
            max-height: 250px;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)
# Use cache_resource to maintain the singleton across reruns
@st.cache_resource
def get_service():
    return MapperService.get_instance()

service = get_service()

st.title("Drone Mapper Engine")

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
    
    with st.expander("Map Settings", expanded=False):
        map_res = st.number_input("Resolution (m/px)", value=service.map_resolution, 
                                  min_value=0.1, max_value=5.0, step=0.1, key="map_res",
                                  help="Lower = more detail but more memory. Requires Reset.")
        band_num = st.number_input("Blend Bands", value=service.band_num, 
                                    min_value=1, max_value=5, step=1,
                                    help="Pyramid levels for multi-band blending. More = smoother seams.")
        tile_size = st.selectbox("Tile Size (px)", [256, 512, 1024], 
                                 index=[256, 512, 1024].index(service.tile_size),
                                 help="Larger tiles = fewer chunks but more memory per tile.")

    if st.button("Reset Map", type="primary"):
        service.reset_map(
            resolution=st.session_state.get("map_res", 0.5),
            band_num=band_num,
            tile_size=tile_size
        )
        st.session_state.processed_files = set()
        st.success("Map reset!")
        st.rerun()

    st.divider()
    st.header("Visualization")
    show_flight_path = st.checkbox("Show Flight Path", value=service.show_flight_path,
                                    help="Overlay drone positions and flight path on the map.")
    service.show_flight_path = show_flight_path

    st.divider()
    st.header("Processing Mode")
    mode = st.radio(
        "Select mode",
        ["Raw GPS", "Pose Graph Optimization", "A/B Comparison"],
        index=2 if service.comparison_mode else (1 if service.enable_pose_graph else 0),
        help="Raw GPS: no correction. Pose Graph: optimized positions. A/B: side-by-side comparison.",
        label_visibility="collapsed"
    )
    service.enable_pose_graph = mode in ["Pose Graph Optimization", "A/B Comparison"]
    service.comparison_mode = mode == "A/B Comparison"
    comparison_mode = service.comparison_mode

    if service.enable_pose_graph or comparison_mode:
        stats = service.pose_optimizer.get_stats()
        if stats['n_frames'] > 0:
            st.caption(f"Edges: {stats['n_edges']} | Max corr: {stats['max_correction']:.2f}m | Avg: {stats['avg_correction']:.2f}m")

    with st.expander("Optimizer Tuning", expanded=False):
        pg_w_gps = st.slider("GPS Weight", 0.1, 5.0, service.pg_w_gps, 0.1,
                              help="Higher = trust GPS more. Lower = trust feature matches more.")
        pg_w_match = st.slider("Match Weight", 0.1, 10.0, service.pg_w_match, 0.1,
                                help="Weight for pairwise feature-match constraints.")
        pg_min_matches = st.slider("Min Matches", 5, 50, service.pg_min_matches, 1,
                                    help="Minimum good SIFT matches to accept a frame pair.")
        pg_iou_thresh = st.slider("IoU Threshold", 0.01, 0.5, service.pg_iou_threshold, 0.01,
                                   help="Minimum overlap to consider frames as neighbors.")
        if st.button("Apply Optimizer Settings"):
            service.update_optimizer_params(pg_w_gps, pg_w_match, pg_min_matches, pg_iou_thresh)
            st.success("Optimizer updated")

    st.divider()
    st.header("Simulation")
    sim_speed = st.slider("Speed (sec)", 0.1, 2.0, 0.5)
    
    if st.button("Run Simulation ▶️"):
        st.session_state.is_simulating = True
        if comparison_mode:
            service._start_comparison()

import plotly.express as px
import plotly.graph_objects as go

# Main Area
# --- MAIN LAYOUT ---

def render_map_figure(map_bytes, title=None):
    """Convert map bytes to a Plotly figure (used for final interactive view)."""
    if not map_bytes:
        return None
    image = Image.open(io.BytesIO(map_bytes))
    img_array = np.array(image)
    fig = px.imshow(img_array)
    fig.update_layout(
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        dragmode="pan",
        height=500,
        title=dict(text=title) if title else None,
    )
    return fig

def display_map_image(placeholder, map_bytes, caption=None):
    """Display map as a static st.image (fast, no flicker). Used during simulation."""
    if map_bytes:
        placeholder.image(map_bytes, caption=caption, use_container_width=True)
    else:
        placeholder.info("No data yet")

is_simulating = st.session_state.get("is_simulating", False)

# Counter for unique plotly keys across multiple update_map_display calls
if "_map_render_id" not in st.session_state:
    st.session_state._map_render_id = 0

if service.comparison_mode:
    # --- SIDE-BY-SIDE VIEW ---
    map_col1, map_col2 = st.columns(2)
    map_ph_raw = map_col1.empty()
    map_ph_opt = map_col2.empty()
    
    def update_map_display(live=False):
        st.session_state._map_render_id += 1
        rid = st.session_state._map_render_id
        raw_bytes = service.get_map_image('raw')
        opt_bytes = service.get_map_image('optimized')
        if is_simulating or live:
            # Fast static images during simulation/upload (no flicker)
            display_map_image(map_ph_raw, raw_bytes, "Raw GPS")
            display_map_image(map_ph_opt, opt_bytes, "Pose Graph Optimized")
        else:
            # Interactive Plotly after simulation completes
            fig_raw = render_map_figure(raw_bytes, "Raw GPS")
            fig_opt = render_map_figure(opt_bytes, "Pose Graph Optimized")
            if fig_raw:
                map_ph_raw.plotly_chart(fig_raw, use_container_width=True, key=f"raw_{rid}")
            else:
                map_ph_raw.info("No data yet")
            if fig_opt:
                map_ph_opt.plotly_chart(fig_opt, use_container_width=True, key=f"opt_{rid}")
            else:
                map_ph_opt.info("No data yet")
else:
    # --- SINGLE MAP VIEW ---
    map_placeholder = st.empty()
    
    def update_map_display(live=False):
        st.session_state._map_render_id += 1
        rid = st.session_state._map_render_id
        map_bytes = service.get_map_image()
        if is_simulating or live:
            display_map_image(map_placeholder, map_bytes)
        else:
            fig = render_map_figure(map_bytes)
            if fig:
                map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_{rid}")
            else:
                map_placeholder.info("No map data yet. Upload images or Run Simulation to start.")

# Initial map render
update_map_display()

st.divider()

# --- METRICS DASHBOARD ---
def show_metrics_dashboard():
    """Display quantitative metrics from the metrics log."""
    logs = service.metrics_log
    if not logs or len(logs) < 2:
        return
    
    st.subheader("📊 Quantitative Metrics")
    
    # Summary stats in columns
    corrections = [m.get('correction_mag', 0) for m in logs]
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Frames", len(logs))
    m_col2.metric("Avg Correction", f"{np.mean(corrections):.2f} m")
    m_col3.metric("Max Correction", f"{np.max(corrections):.2f} m")
    
    pg_stats = service.pose_optimizer.get_stats()
    m_col4.metric("Feature Edges", pg_stats['n_edges'])
    
    # Comparison-mode metrics
    if service.comparison_mode:
        ssim_raw = [m.get('seam_ssim_raw', 0) for m in logs if 'seam_ssim_raw' in m]
        ssim_opt = [m.get('seam_ssim_opt', 0) for m in logs if 'seam_ssim_opt' in m]
        
        if ssim_raw and ssim_opt:
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Seam SSIM (Raw GPS)", f"{np.mean(ssim_raw):.4f}")
            s_col2.metric("Seam SSIM (Optimized)", f"{np.mean(ssim_opt):.4f}")
            improvement = (np.mean(ssim_opt) - np.mean(ssim_raw))
            s_col3.metric("SSIM Improvement", f"{improvement:+.4f}", 
                         delta=f"{improvement:+.4f}",
                         delta_color="normal")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Per-frame correction magnitude
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(
            x=list(range(len(corrections))),
            y=corrections,
            marker_color='steelblue',
            name='Correction'
        ))
        fig_corr.update_layout(
            title="Per-Frame Correction Magnitude",
            xaxis_title="Frame #",
            yaxis_title="Correction (m)",
            height=300,
            margin=dict(l=40, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with chart_col2:
        # GPS positions: raw vs corrected
        gps_x = [m.get('gps_x', 0) for m in logs]
        gps_y = [m.get('gps_y', 0) for m in logs]
        corr_x = [m.get('gps_x', 0) + m.get('correction_x', 0) for m in logs]
        corr_y = [m.get('gps_y', 0) + m.get('correction_y', 0) for m in logs]
        
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Scatter(
            x=gps_y, y=gps_x,
            mode='lines+markers',
            name='Raw GPS',
            marker=dict(size=6, color='red'),
            line=dict(color='red', dash='dot'),
        ))
        fig_pos.add_trace(go.Scatter(
            x=corr_y, y=corr_x,
            mode='lines+markers',
            name='Optimized',
            marker=dict(size=6, color='green'),
            line=dict(color='green'),
        ))
        # Draw correction arrows
        for i in range(len(logs)):
            if corrections[i] > 0.01:
                fig_pos.add_annotation(
                    x=corr_y[i], y=corr_x[i],
                    ax=gps_y[i], ay=gps_x[i],
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor='rgba(100,100,100,0.3)',
                )
        fig_pos.update_layout(
            title="Flight Path: Raw GPS vs Optimized",
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            height=300,
            margin=dict(l=40, r=10, t=40, b=40),
            yaxis=dict(scaleanchor="x"),
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    
    # Seam SSIM over time (comparison mode only)
    if service.comparison_mode:
        ssim_raw = [m.get('seam_ssim_raw', None) for m in logs]
        ssim_opt = [m.get('seam_ssim_opt', None) for m in logs]
        frames_with_ssim = [i for i, s in enumerate(ssim_raw) if s is not None]
        
        if frames_with_ssim:
            fig_ssim = go.Figure()
            fig_ssim.add_trace(go.Scatter(
                x=frames_with_ssim,
                y=[ssim_raw[i] for i in frames_with_ssim],
                mode='lines+markers', name='Raw GPS',
                marker=dict(color='red'), line=dict(color='red', dash='dot'),
            ))
            fig_ssim.add_trace(go.Scatter(
                x=frames_with_ssim,
                y=[ssim_opt[i] for i in frames_with_ssim],
                mode='lines+markers', name='Optimized',
                marker=dict(color='green'), line=dict(color='green'),
            ))
            fig_ssim.update_layout(
                title="Seam SSIM Over Time (higher = better blending)",
                xaxis_title="Frame #",
                yaxis_title="Mean Seam SSIM",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_ssim, use_container_width=True)
    
    # Flight path comparison (A/B mode) - both paths on same chart
    if service.comparison_mode and service.mapper_raw is not None and service.mapper_optimized is not None:
        raw_path = service.mapper_raw.flight_path
        opt_path = service.mapper_optimized.flight_path
        
        if len(raw_path) >= 2 and len(opt_path) >= 2:
            st.subheader("\U0001f6e9\ufe0f Flight Path Comparison")
            
            raw_x = [p[0] for p in raw_path]
            raw_y = [p[1] for p in raw_path]
            opt_x = [p[0] for p in opt_path]
            opt_y = [p[1] for p in opt_path]
            
            fig_fp = go.Figure()
            
            # Raw GPS path
            fig_fp.add_trace(go.Scatter(
                x=raw_x, y=raw_y,
                mode='lines+markers',
                name='Raw GPS Path',
                marker=dict(size=6, color='red', symbol='circle'),
                line=dict(color='red', dash='dot', width=2),
            ))
            
            # Optimized path
            fig_fp.add_trace(go.Scatter(
                x=opt_x, y=opt_y,
                mode='lines+markers',
                name='Optimized Path',
                marker=dict(size=6, color='#00CC66', symbol='diamond'),
                line=dict(color='#00CC66', width=2),
            ))
            
            # Displacement arrows between corresponding points
            for i in range(min(len(raw_path), len(opt_path))):
                dx = opt_x[i] - raw_x[i]
                dy = opt_y[i] - raw_y[i]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0.05:  # Only show if shift > 5cm
                    fig_fp.add_annotation(
                        x=opt_x[i], y=opt_y[i],
                        ax=raw_x[i], ay=raw_y[i],
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1,
                        arrowcolor='rgba(100,100,100,0.4)',
                    )
            
            # Mark start and end
            fig_fp.add_trace(go.Scatter(
                x=[raw_x[0], raw_x[-1]], y=[raw_y[0], raw_y[-1]],
                mode='markers+text',
                name='Start / End',
                marker=dict(size=12, color=['green', 'red'], symbol='star'),
                text=['START', 'END'],
                textposition='top center',
                textfont=dict(size=10, color='white'),
                showlegend=False,
            ))
            
            # Per-point displacement stats
            displacements = [np.sqrt((opt_x[i]-raw_x[i])**2 + (opt_y[i]-raw_y[i])**2) 
                            for i in range(min(len(raw_path), len(opt_path)))]
            
            fig_fp.update_layout(
                title=f"Flight Path: Raw GPS vs Optimized (avg shift: {np.mean(displacements):.2f}m, max: {np.max(displacements):.2f}m)",
                xaxis_title="East (m)",
                yaxis_title="North (m)" ,
                height=400,
                margin=dict(l=40, r=10, t=50, b=40),
                yaxis=dict(scaleanchor="x"),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
                plot_bgcolor='rgba(20,20,30,1)',
                paper_bgcolor='rgba(20,20,30,1)',
                font=dict(color='white'),
            )
            st.plotly_chart(fig_fp, use_container_width=True)
            
            # Displacement magnitude per frame
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Bar(
                x=list(range(len(displacements))),
                y=displacements,
                marker_color=['#FF6B6B' if d > np.mean(displacements) else '#4ECDC4' for d in displacements],
                name='Displacement',
            ))
            fig_disp.add_hline(y=np.mean(displacements), line_dash="dash", line_color="yellow",
                              annotation_text=f"avg: {np.mean(displacements):.2f}m")
            fig_disp.update_layout(
                title="Per-Frame Position Displacement (Raw vs Optimized)",
                xaxis_title="Frame #",
                yaxis_title="Displacement (m)",
                height=250,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_disp, use_container_width=True)

# Section 2: Logs and Info (Bottom)
b_col1, b_col2 = st.columns([1, 1])

with b_col1:
    with st.expander("System Status", expanded=True):
        tile_count = len(service.mapper.tiles)
        path_len = len(service.mapper.flight_path)
        st.markdown(f"""
        **Operational Status**
        - **Engine:** \u2705 Online
        - **Drone Connection:** \u26A0\uFE0F Disconnected (Simulation Mode)
        - **Map Resolution:** {service.map_resolution} m/px
        - **Blend Bands:** {service.band_num} | **Tile Size:** {service.tile_size}px
        - **Active Tile Chunks:** {tile_count}
        - **Flight Path Points:** {path_len}
        """)
        if st.session_state.get("is_simulating"):
            st.info("\u25B6\uFE0F Simulation Running...")
        else:
            st.success("System Ready for Input")

with b_col2:
    # st.subheader("Processing Log")
    # Use expander to save space, with scrollable container
    with st.expander("Processing Log", expanded=True):
        log_container = st.container(height=250)
    
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
                update_map_display(live=True)
                
                time.sleep(sim_speed)
                progress_bar.progress((i + 1) / len(images))
            
            st.session_state.is_simulating = False
            status_text.text("Simulation Complete")
            
            # Show metrics after simulation
            show_metrics_dashboard()
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
                
                # Update map after each upload processing (fast static image)
                update_map_display(live=True)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Brief yield to keep WebSocket alive on hosted environments
                time.sleep(0.1)
        
        status_text.text("Ready")
        # Final interactive Plotly render after all uploads processed
        update_map_display()

st.divider()

# Always show metrics dashboard if data exists
show_metrics_dashboard()

# --- PERFORMANCE DASHBOARD ---
def show_performance_dashboard():
    """Display per-frame performance timing and resource usage."""
    perf_summary = service.get_performance_summary()
    if perf_summary is None:
        return
    
    st.subheader("\u2699\ufe0f Performance")
    
    # Top-level KPIs
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Frames", perf_summary['n_frames'])
    p2.metric("Throughput", f"{perf_summary['fps']:.2f} fps")
    p3.metric("Avg Frame Time", f"{perf_summary['avg_total_ms']:.0f} ms")
    p4.metric("Tiles", perf_summary['tile_count'])
    p5.metric("Tile Memory", f"{perf_summary['tile_memory_mb']:.1f} MB")
    
    # Timing breakdown
    with st.expander("Timing Breakdown", expanded=False):
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            # Stacked bar / breakdown
            labels = ['Pose Extract', 'Image Decode', 'Preprocessing', 'Optimization', 'Map Feed']
            values = [
                perf_summary['avg_pose_ms'],
                perf_summary['avg_decode_ms'],
                np.mean([p.get('preprocessing_ms', 0) for p in service.perf_log]),
                perf_summary['avg_optimize_ms'],
                perf_summary['avg_feed_ms'],
            ]
            fig_breakdown = go.Figure()
            fig_breakdown.add_trace(go.Bar(
                x=labels, y=values,
                marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'],
                text=[f"{v:.1f}" for v in values],
                textposition='auto',
            ))
            fig_breakdown.update_layout(
                title="Avg Time per Stage (ms)",
                yaxis_title="ms",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        with t_col2:
            # Per-frame total time over time
            frame_times = [p['total_ms'] for p in service.perf_log]
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=list(range(len(frame_times))),
                y=frame_times,
                mode='lines+markers',
                name='Total',
                marker=dict(size=4, color='steelblue'),
                line=dict(color='steelblue'),
            ))
            # Add feed time line
            feed_times = [p.get('feed_ms', p.get('feed_raw_ms', 0)) for p in service.perf_log]
            fig_timeline.add_trace(go.Scatter(
                x=list(range(len(feed_times))),
                y=feed_times,
                mode='lines',
                name='Map Feed',
                line=dict(color='#FFA15A', dash='dot'),
            ))
            fig_timeline.update_layout(
                title="Per-Frame Processing Time",
                xaxis_title="Frame #",
                yaxis_title="ms",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Resource usage over time
        mem_vals = [p.get('tile_memory_mb', 0) for p in service.perf_log]
        tile_counts = [p.get('tile_count', 0) for p in service.perf_log]
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=list(range(len(mem_vals))), y=mem_vals,
            mode='lines+markers', name='Tile Memory (MB)',
            marker=dict(size=4, color='#AB63FA'),
        ))
        fig_res.add_trace(go.Scatter(
            x=list(range(len(tile_counts))), y=tile_counts,
            mode='lines+markers', name='Tile Count',
            yaxis='y2',
            marker=dict(size=4, color='#00CC96'),
        ))
        fig_res.update_layout(
            title="Resource Usage Over Time",
            xaxis_title="Frame #",
            yaxis=dict(title="Memory (MB)"),
            yaxis2=dict(title="Tiles", overlaying='y', side='right'),
            height=280,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_res, use_container_width=True)
        
        # Summary table
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Elapsed | {perf_summary['elapsed_sec']:.1f} s |
        | Peak Frame Time | {perf_summary['peak_total_ms']:.0f} ms |
        | Total Image Data | {perf_summary['total_images_mb']:.1f} MB |
        | Avg Pose Extraction | {perf_summary['avg_pose_ms']:.1f} ms |
        | Avg Image Decode | {perf_summary['avg_decode_ms']:.1f} ms |
        | Avg Optimization | {perf_summary['avg_optimize_ms']:.1f} ms |
        | Avg Map Feed | {perf_summary['avg_feed_ms']:.1f} ms |
        """)

show_performance_dashboard()

st.caption("2026 - Jugal Alan - Powered by Streamlit")
