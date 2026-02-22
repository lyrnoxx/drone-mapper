import cv2
import numpy as np
import time
import sys
from .core import MultiBandMap2D, PoseExtractor, PoseGraphOptimizer

class MapperService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MapperService()
        return cls._instance

    def __init__(self):
        # DJI M3T camera parameters (approximate)
        self.focal_len_mm = 24.0
        self.sensor_width_mm = 36.0
        self.img_width_px = 4000
        self.update_parameters(self.focal_len_mm, self.sensor_width_mm, self.img_width_px)
        
        # Configurable map parameters
        self.map_resolution = 0.5
        self.band_num = 2
        self.tile_size = 512
        
        # Initialize engine and pose extractor
        self.mapper = MultiBandMap2D(resolution=self.map_resolution, band_num=self.band_num, tile_size=self.tile_size)
        self.pose_extractor = PoseExtractor()
        
        # Pose graph optimization (configurable weights)
        self.pg_w_gps = 1.0
        self.pg_w_match = 2.0
        self.pg_min_matches = 15
        self.pg_iou_threshold = 0.10
        self.pose_optimizer = PoseGraphOptimizer(
            w_gps=self.pg_w_gps, w_match=self.pg_w_match,
            min_matches=self.pg_min_matches, iou_threshold=self.pg_iou_threshold
        )
        self.enable_pose_graph = False
        
        # Comparison mode: run both raw GPS and optimized in parallel
        self.comparison_mode = False
        self.mapper_raw = None        # Raw GPS mapper (no optimization)
        self.mapper_optimized = None  # Pose-graph-optimized mapper
        
        # Flight path overlay toggle
        self.show_flight_path = True
        
        # Per-frame metrics log
        self.metrics_log = []  # List of dicts per frame
        
        # Performance tracking
        self.perf_log = []  # List of per-frame timing dicts
        self.session_start_time = time.time()

    def update_parameters(self, focal_len_mm, sensor_width_mm, img_width_px):
        """Update camera intrinsic parameters."""
        fx = img_width_px * (focal_len_mm / sensor_width_mm)
        fy = fx
        cx = 2000 # Raw 4000/2
        cy = 1500 # Raw 3000/2
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return fx
        
    def _start_comparison(self):
        """Initialize dual mappers for comparison mode."""
        self.mapper_raw = MultiBandMap2D(resolution=self.map_resolution, band_num=self.band_num, tile_size=self.tile_size)
        self.mapper_optimized = MultiBandMap2D(resolution=self.map_resolution, band_num=self.band_num, tile_size=self.tile_size)
        self.metrics_log = []
        self.perf_log = []
        self.pose_optimizer.reset()
    
    # Maximum image dimension for processing (prevents timeout on hosted envs)
    MAX_IMG_DIM = 2000

    def process_image(self, img_bytes):
        """Process a raw image: extract pose -> optimize -> feed to mapper."""
        t_total_start = time.time()
        perf = {}
        
        # Extract pose
        t0 = time.time()
        pose_matrix = self.pose_extractor.extract_pose(img_bytes)
        perf['pose_extraction_ms'] = (time.time() - t0) * 1000
        
        if pose_matrix is None:
            return {"status": "error", "message": "Could not extract pose from EXIF"}
            
        # Decode image
        t0 = time.time()
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
        perf['image_decode_ms'] = (time.time() - t0) * 1000
        perf['image_size_bytes'] = len(img_bytes)
        
        if frame is None:
             return {"status": "error", "message": "Failed to decode image"}
        
        # Downscale large images to prevent timeout on hosted environments
        h, w = frame.shape[:2]
        perf['image_resolution'] = f"{w}x{h}"
        K_frame = self.K.copy()
        if max(h, w) > self.MAX_IMG_DIM:
            scale = self.MAX_IMG_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            K_frame[0, 0] *= scale; K_frame[1, 1] *= scale
            K_frame[0, 2] *= scale; K_frame[1, 2] *= scale
            perf['downscaled_to'] = f"{frame.shape[1]}x{frame.shape[0]}"
        raw_pose = pose_matrix.copy()
        
        # Prepare downscaled gray for SIFT (used by optimizer and/or metrics)
        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
        perf['preprocessing_ms'] = (time.time() - t0) * 1000
        K_half = K_frame.copy()
        K_half[0, 0] /= 2; K_half[1, 1] /= 2
        K_half[0, 2] /= 2; K_half[1, 2] /= 2
        
        # Build frame metrics
        frame_idx = len(self.metrics_log)
        metric = {
            'frame': frame_idx,
            'gps_x': raw_pose[0, 3],
            'gps_y': raw_pose[1, 3],
            'altitude': -raw_pose[2, 3],
        }
        
        if self.comparison_mode:
            # Ensure comparison mappers are initialized
            if self.mapper_raw is None or self.mapper_optimized is None:
                self._start_comparison()
            # --- COMPARISON MODE: feed both mappers ---
            # 1. Raw GPS mapper
            t0 = time.time()
            self.mapper_raw.feed(frame, raw_pose, K_frame)
            perf['feed_raw_ms'] = (time.time() - t0) * 1000
            metric['time_raw'] = perf['feed_raw_ms'] / 1000
            
            # 2. Optimized mapper  
            t0 = time.time()
            optimized_pose = self.pose_optimizer.add_frame(gray_small, pose_matrix, K_half)
            perf['optimization_ms'] = (time.time() - t0) * 1000
            
            t0 = time.time()
            self.mapper_optimized.feed(frame, optimized_pose, K_frame)
            perf['feed_optimized_ms'] = (time.time() - t0) * 1000
            metric['time_optimized'] = (perf['optimization_ms'] + perf['feed_optimized_ms']) / 1000
            
            # Correction magnitude
            corr = optimized_pose[:3, 3] - raw_pose[:3, 3]
            metric['correction_x'] = corr[0]
            metric['correction_y'] = corr[1]
            metric['correction_mag'] = np.linalg.norm(corr[:2])
            
            # Number of edges so far
            stats = self.pose_optimizer.get_stats()
            metric['n_edges'] = stats['n_edges']
            metric['n_inliers'] = stats.get('n_edges', 0)  # edges as proxy
            
            # Overlap seam quality (SSIM at overlap region)
            if frame_idx > 0:
                metric['seam_ssim_raw'] = self._compute_seam_ssim(self.mapper_raw)
                metric['seam_ssim_opt'] = self._compute_seam_ssim(self.mapper_optimized)
            
            # Also update the main mapper to optimized for primary display
            self.mapper = self.mapper_optimized
            
        elif self.enable_pose_graph:
            # Single optimized pipeline
            t0 = time.time()
            pose_matrix = self.pose_optimizer.add_frame(gray_small, pose_matrix, K_half)
            perf['optimization_ms'] = (time.time() - t0) * 1000
            metric['optimization_time'] = perf['optimization_ms'] / 1000
            
            corr = pose_matrix[:3, 3] - raw_pose[:3, 3]
            metric['correction_x'] = corr[0]
            metric['correction_y'] = corr[1]
            metric['correction_mag'] = np.linalg.norm(corr[:2])
            
            t0 = time.time()
            self.mapper.feed(frame, pose_matrix, K_frame)
            perf['feed_ms'] = (time.time() - t0) * 1000
        else:
            # Raw GPS only
            t0 = time.time()
            self.mapper.feed(frame, pose_matrix, K_frame)
            perf['feed_ms'] = (time.time() - t0) * 1000
            metric['correction_mag'] = 0.0
        
        # Total time
        perf['total_ms'] = (time.time() - t_total_start) * 1000
        perf['frame'] = len(self.metrics_log)
        perf['tile_count'] = len(self.mapper.tiles)
        perf['tile_memory_mb'] = self.mapper.get_memory_usage() / (1024 * 1024)
        
        self.perf_log.append(perf)
        self.metrics_log.append(metric)
        
        return {"status": "success", "message": "Image processed"}
    
    def _compute_seam_ssim(self, mapper):
        """Compute mean SSIM across tile boundaries as a seam quality metric."""
        try:
            img = mapper.render_map(0)
            if img is None or img.shape[0] < 100 or img.shape[1] < 100:
                return 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Compute gradient magnitude (edges/seams show as high gradient)
            # Lower gradient variance at tile boundaries = better blending
            ts = mapper.tile_size
            ssim_scores = []
            
            # Check horizontal seams
            for y in range(ts, gray.shape[0] - 10, ts):
                if y + 10 > gray.shape[0]:
                    break
                strip_above = gray[max(0, y-5):y, :].astype(np.float32)
                strip_below = gray[y:min(gray.shape[0], y+5), :].astype(np.float32)
                if strip_above.shape == strip_below.shape and strip_above.size > 0:
                    # Normalized cross-correlation as quick SSIM proxy
                    mu1, mu2 = strip_above.mean(), strip_below.mean()
                    if mu1 > 1 and mu2 > 1:  # skip black regions
                        s1, s2 = strip_above.std(), strip_below.std()
                        cov = np.mean((strip_above - mu1) * (strip_below - mu2))
                        C1, C2 = 6.5025, 58.5225  # SSIM constants
                        ssim = ((2*mu1*mu2 + C1) * (2*cov + C2)) / \
                               ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2))
                        ssim_scores.append(float(ssim))
            
            # Check vertical seams
            for x in range(ts, gray.shape[1] - 10, ts):
                if x + 10 > gray.shape[1]:
                    break
                strip_left = gray[:, max(0, x-5):x].astype(np.float32)
                strip_right = gray[:, x:min(gray.shape[1], x+5)].astype(np.float32)
                if strip_left.shape == strip_right.shape and strip_left.size > 0:
                    mu1, mu2 = strip_left.mean(), strip_right.mean()
                    if mu1 > 1 and mu2 > 1:
                        s1, s2 = strip_left.std(), strip_right.std()
                        cov = np.mean((strip_left - mu1) * (strip_right - mu2))
                        C1, C2 = 6.5025, 58.5225
                        ssim = ((2*mu1*mu2 + C1) * (2*cov + C2)) / \
                               ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2))
                        ssim_scores.append(float(ssim))
            
            return float(np.mean(ssim_scores)) if ssim_scores else 0.0
        except Exception as e:
            print(f"SSIM Error: {e}")
            return 0.0

    def get_map_image(self, which='main'):
        """Return a map as encoded JPEG buffer.
        Args:
            which: 'main', 'raw', or 'optimized'
        """
        show_fp = self.show_flight_path
        if which == 'raw' and self.mapper_raw is not None:
            img = self.mapper_raw.render_map(0, show_flight_path=show_fp)
        elif which == 'optimized' and self.mapper_optimized is not None:
            img = self.mapper_optimized.render_map(0, show_flight_path=show_fp)
        else:
            img = self.mapper.render_map(0, show_flight_path=show_fp)
        
        if img is None:
            return None
        
        success, buffer = cv2.imencode('.jpg', img)
        if not success:
            return None
        return buffer.tobytes()

    def reset_map(self, resolution=None, band_num=None, tile_size=None):
        if resolution is not None:
            self.map_resolution = resolution
        if band_num is not None:
            self.band_num = band_num
        if tile_size is not None:
            self.tile_size = tile_size
        self.mapper = MultiBandMap2D(resolution=self.map_resolution, band_num=self.band_num, tile_size=self.tile_size)
        self.mapper_raw = None
        self.mapper_optimized = None
        self.pose_optimizer.reset()
        self.pose_extractor = PoseExtractor()
        self.metrics_log = []
        self.perf_log = []
        self.session_start_time = time.time()
        return True
    
    def update_optimizer_params(self, w_gps=None, w_match=None, min_matches=None, iou_threshold=None):
        """Update pose graph optimizer weights (takes effect on next reset)."""
        if w_gps is not None:
            self.pg_w_gps = w_gps
        if w_match is not None:
            self.pg_w_match = w_match
        if min_matches is not None:
            self.pg_min_matches = min_matches
        if iou_threshold is not None:
            self.pg_iou_threshold = iou_threshold
        self.pose_optimizer = PoseGraphOptimizer(
            w_gps=self.pg_w_gps, w_match=self.pg_w_match,
            min_matches=self.pg_min_matches, iou_threshold=self.pg_iou_threshold
        )
    
    def get_performance_summary(self):
        """Return an aggregate performance summary."""
        if not self.perf_log:
            return None
        
        total_times = [p['total_ms'] for p in self.perf_log]
        feed_times = [p.get('feed_ms', p.get('feed_raw_ms', 0)) for p in self.perf_log]
        pose_times = [p.get('pose_extraction_ms', 0) for p in self.perf_log]
        decode_times = [p.get('image_decode_ms', 0) for p in self.perf_log]
        opt_times = [p.get('optimization_ms', 0) for p in self.perf_log]
        
        latest = self.perf_log[-1]
        elapsed = time.time() - self.session_start_time
        
        return {
            'n_frames': len(self.perf_log),
            'elapsed_sec': elapsed,
            'fps': len(self.perf_log) / elapsed if elapsed > 0 else 0,
            'avg_total_ms': np.mean(total_times),
            'avg_feed_ms': np.mean(feed_times),
            'avg_pose_ms': np.mean(pose_times),
            'avg_decode_ms': np.mean(decode_times),
            'avg_optimize_ms': np.mean(opt_times) if any(t > 0 for t in opt_times) else 0,
            'last_total_ms': latest['total_ms'],
            'tile_count': latest.get('tile_count', 0),
            'tile_memory_mb': latest.get('tile_memory_mb', 0),
            'peak_total_ms': max(total_times),
            'total_images_mb': sum(p.get('image_size_bytes', 0) for p in self.perf_log) / (1024 * 1024),
        }
