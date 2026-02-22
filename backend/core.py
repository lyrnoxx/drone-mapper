import cv2
import numpy as np
import threading
import re
import logging
import os
from pyproj import Proj
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

logger = logging.getLogger(__name__)

# ================= POSE EXTRACTION FROM EXIF =================
class PoseExtractor:
    """Extract camera pose from DJI image XMP metadata (pure Python, no exiftool)."""
    
    def __init__(self):
        self.proj = None
        self.origin_x = None
        self.origin_y = None
        self.initialized = False
    
    def get_meta(self, img_bytes):
        """
        Extract DJI metadata directly from JPEG XMP data.
        
        DJI stores gimbal angles and GPS in XMP-drone-dji namespace.
        We search for the XMP packet in the JPEG bytes and parse it.
        """
        # Find XMP data in JPEG - it's between <?xpacket begin and <?xpacket end
        # Or we can look for the drone-dji namespace
        try:
            # Convert bytes to string for regex (XMP is UTF-8 text)
            # Find the XMP segment - typically starts with http://ns.adobe.com/xap/1.0/
            xmp_start = img_bytes.find(b'<x:xmpmeta')
            xmp_end = img_bytes.find(b'</x:xmpmeta>')
            
            if xmp_start == -1 or xmp_end == -1:
                # Try alternative XMP markers
                xmp_start = img_bytes.find(b'<?xpacket begin')
                xmp_end = img_bytes.find(b'<?xpacket end')
            
            if xmp_start == -1 or xmp_end == -1:
                print("Warning: No XMP data found in image")
                return None
            
            xmp_data = img_bytes[xmp_start:xmp_end + 50].decode('utf-8', errors='ignore')
            
            # Extract DJI-specific fields using regex
            # Format: drone-dji:FieldName="value" or drone-dji:FieldName="+value"
            def extract_value(field_name):
                # Try attribute format: drone-dji:FieldName="value"
                # Use re.IGNORECASE to handle GPSLatitude vs GpsLatitude
                pattern = rf'drone-dji:{field_name}="([^"]*)"'
                match = re.search(pattern, xmp_data, re.IGNORECASE)
                if match:
                    return match.group(1)
                
                # Try element format: <drone-dji:FieldName>value</drone-dji:FieldName>
                pattern = rf'<drone-dji:{field_name}>([^<]*)</drone-dji:{field_name}>'
                match = re.search(pattern, xmp_data, re.IGNORECASE)
                if match:
                    return match.group(1)
                
                return None
            
            # Extract required fields
            data = {}
            
            # GPS coordinates (DJI uses GpsLatitude not GPSLatitude)
            gps_lat = extract_value('GpsLatitude')
            gps_lon = extract_value('GpsLongitude')
            rel_alt = extract_value('RelativeAltitude')
            
            # Gimbal angles
            gimbal_roll = extract_value('GimbalRollDegree')
            gimbal_pitch = extract_value('GimbalPitchDegree')
            gimbal_yaw = extract_value('GimbalYawDegree')
            
            if gps_lat:
                data['GPSLatitude'] = float(gps_lat.replace('+', ''))
            if gps_lon:
                data['GPSLongitude'] = float(gps_lon.replace('+', ''))
            if rel_alt:
                data['RelativeAltitude'] = float(rel_alt.replace('+', ''))
            if gimbal_roll:
                data['GimbalRollDegree'] = float(gimbal_roll.replace('+', ''))
            if gimbal_pitch:
                data['GimbalPitchDegree'] = float(gimbal_pitch.replace('+', ''))
            if gimbal_yaw:
                data['GimbalYawDegree'] = float(gimbal_yaw.replace('+', ''))
            
            return data
            
        except Exception as e:
            print(f"Error parsing XMP data: {e}")
            return None
    
    def compute_camera_pose(self, metadata):
        """
        Compute camera pose from DJI gimbal angles.
        
        Standard Coordinate Mappings (NED: X=North, Y=East, Z=Down):
        Base orientation (Yaw=0, Pitch=0, Roll=0): Camera looks North.
        CV +Z (Forward) -> NED North (+X)
        CV +X (Right) -> NED East (+Y)
        CV +Y (Down) -> NED Down (+Z)
        """
        yaw = metadata['GimbalYawDegree']
        pitch = metadata['GimbalPitchDegree']
        roll = metadata['GimbalRollDegree']
        
        # Base transformation from CV to NED
        R_base = R.from_matrix([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        # DJI Pitch: 0 = horizon, -90 = nadir.
        # In NED, rotating around Y (East) by -90 deg moves X (North) to Z (Down).
        # So we use pitch directly.
        R_y = R.from_euler('y', pitch, degrees=True)
        R_z = R.from_euler('z', yaw, degrees=True)
        R_x = R.from_euler('x', roll, degrees=True)
        
        # Combined rotation: Yaw(Z) * Pitch(Y) * Roll(X) * Base
        return R_z * R_y * R_x * R_base
    
    def extract_pose(self, img_bytes):
        """
        Extract full camera pose (4x4 matrix) from image bytes.
        
        Returns:
            pose_matrix: 4x4 camera pose in NED frame, or None if extraction fails
        """
        try:
            m = self.get_meta(img_bytes)
            
            # Check required fields
            required = ['GPSLatitude', 'GPSLongitude', 'RelativeAltitude',
                       'GimbalRollDegree', 'GimbalPitchDegree', 'GimbalYawDegree']
            if not all(k in m for k in required):
                print("Warning: Missing required EXIF fields")
                return None
            
            # Initialize projection on first image
            if not self.initialized:
                utm_zone = int((m['GPSLongitude'] + 180) / 6) + 1
                self.proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
                # Proj returns (east, north). We want North=X, East=Y for NED.
                self.origin_y, self.origin_x = self.proj(m['GPSLongitude'], m['GPSLatitude'])
                self.initialized = True
                print(f"UTM Zone: {utm_zone}")
                print(f"Origin (N, E): ({self.origin_x:.2f}, {self.origin_y:.2f})")
            
            # Convert GPS to UTM
            # pyproj Proj returns (east, north)
            ty, tx = self.proj(m['GPSLongitude'], m['GPSLatitude'])
            
            # Make relative to origin (NED: X=North, Y=East)
            tx_rel = tx - self.origin_x
            ty_rel = ty - self.origin_y
            tz_rel = -m['RelativeAltitude']
            
            # Compute camera rotation
            R_cam = self.compute_camera_pose(m)
            
            # Build 4x4 pose matrix
            T = np.eye(4)
            T[:3, :3] = R_cam.as_matrix()
            T[:3, 3] = [tx_rel, ty_rel, tz_rel]
            
            self.last_metadata = m
            return T
            
        except Exception as e:
            print(f"Error extracting pose: {e}")
            return None


# ================= POSE GRAPH OPTIMIZER =================
class PoseGraphOptimizer:
    """
    Translation-only block adjustment via sparse least-squares.
    
    Matches SIFT features between overlapping frame pairs to compute
    relative translation constraints, then jointly solves for corrected
    positions that satisfy both GPS priors and pairwise constraints.
    
    Formulation:
        min_{t_i} sum_i w_gps * ||t_i - t_i^gps||^2
                + sum_{(i,j)} w_match * ||t_j - t_i - delta_ij||^2
    
    This reduces to a sparse linear system Ax = b, solved via LSQR.
    Rotations from the DJI IMU are trusted and not adjusted.
    """
    
    def __init__(self, w_gps=1.0, w_match=2.0, min_matches=15, iou_threshold=0.10):
        """
        Args:
            w_gps: Weight for GPS position priors
            w_match: Weight for pairwise feature-match constraints  
            min_matches: Minimum good SIFT matches to accept a pair
            iou_threshold: Minimum IoU to consider two frames as overlapping
        """
        self.w_gps = w_gps
        self.w_match = w_match
        self.min_matches = min_matches
        self.iou_threshold = iou_threshold
        
        # Feature detector
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )
        
        # Accumulated data
        self.frames = []       # List of {gray, pose, gps_xy, corners, keypoints, descriptors}
        self.edges = []        # List of (i, j, delta_xy) pairwise constraints
        self.corrections = []  # Current correction offsets [dx, dy] per frame
        
        self._dirty = False    # Whether re-solve is needed
    
    def _compute_ground_corners(self, pose, K):
        """Compute ground-plane footprint corners from pose + intrinsics."""
        # Approximate using a standard 4000x3000 frame
        h, w = 3000, 4000
        u = np.array([0, w-1, w-1, 0], np.float32)
        v = np.array([0, 0, h-1, h-1], np.float32)
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        ray_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=1)
        ray_cam /= np.linalg.norm(ray_cam, axis=1, keepdims=True)
        
        Rot = pose[:3, :3]
        t = pose[:3, 3]
        ray_world = (Rot @ ray_cam.T).T
        
        lams = -t[2] / ray_world[:, 2]
        if np.any(lams < 0):
            return None
        
        pts_world = t + lams[:, None] * ray_world
        # Map to display frame: East = +X, North = +Y
        corners = np.zeros((4, 2), np.float32)
        corners[:, 0] = pts_world[:, 1]   # Easting
        corners[:, 1] = -pts_world[:, 0]  # -Northing (north up)
        return corners
    
    def _compute_iou(self, corners1, corners2):
        """Fast rasterized IoU between two quadrilaterals."""
        if corners1 is None or corners2 is None:
            return 0.0
        try:
            all_pts = np.vstack([corners1, corners2])
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            calc_res = 0.5  # coarser for speed
            w = int((xmax - xmin) / calc_res) + 2
            h = int((ymax - ymin) / calc_res) + 2
            if w > 5000 or h > 5000:
                return 0.0
            mask1 = np.zeros((h, w), dtype=np.uint8)
            mask2 = np.zeros((h, w), dtype=np.uint8)
            p1 = ((corners1 - [xmin, ymin]) / calc_res).astype(np.int32)
            p2 = ((corners2 - [xmin, ymin]) / calc_res).astype(np.int32)
            cv2.fillPoly(mask1, [p1], 255)
            cv2.fillPoly(mask2, [p2], 255)
            inter = cv2.countNonZero(cv2.bitwise_and(mask1, mask2))
            union = cv2.countNonZero(cv2.bitwise_or(mask1, mask2))
            return inter / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def _match_pair(self, idx_i, idx_j, K):
        """
        Match SIFT features between frames i and j.
        If enough matches, compute the relative translation delta_ij
        by unprojecting matched pixels to the ground plane.
        
        Returns (delta_x, delta_y, n_inliers) or None if matching fails.
        """
        fi, fj = self.frames[idx_i], self.frames[idx_j]
        des_i, des_j = fi['descriptors'], fj['descriptors']
        
        if des_i is None or des_j is None:
            return None
        if len(des_i) < 2 or len(des_j) < 2:
            return None
        
        try:
            raw_matches = self.flann.knnMatch(des_i, des_j, k=2)
        except cv2.error:
            return None
        
        # Lowe's ratio test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        if len(good) < self.min_matches:
            return None
        
        # Unproject matched keypoints to ground plane for both frames
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        def unproject_to_ground(kps, matches_indices, pose):
            pts_px = np.float32([kps[m].pt for m in matches_indices])
            x_cam = (pts_px[:, 0] - cx) / fx
            y_cam = (pts_px[:, 1] - cy) / fy
            ray_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=1)
            ray_cam /= np.linalg.norm(ray_cam, axis=1, keepdims=True)
            Rot = pose[:3, :3]
            t = pose[:3, 3]
            ray_world = (Rot @ ray_cam.T).T
            lams = -t[2] / ray_world[:, 2]
            # Filter out invalid rays
            valid = lams > 0
            pts_world = t + lams[:, None] * ray_world
            return pts_world[:, :2], valid  # NED XY
        
        query_indices = [m.queryIdx for m in good]
        train_indices = [m.trainIdx for m in good]
        
        pts_i, valid_i = unproject_to_ground(fi['keypoints'], query_indices, fi['pose'])
        pts_j, valid_j = unproject_to_ground(fj['keypoints'], train_indices, fj['pose'])
        
        valid = valid_i & valid_j
        pts_i, pts_j = pts_i[valid], pts_j[valid]
        
        if len(pts_i) < self.min_matches:
            return None
        
        # The relative translation measured by the matches:
        # If frames were perfectly placed, pts_i[k] == pts_j[k] for all k.
        # The median difference gives a robust estimate of the relative offset.
        deltas = pts_j - pts_i  # How much j needs to shift relative to i
        
        # Use RANSAC-like robust estimation: median
        delta_xy = np.median(deltas, axis=0)
        
        # Compute inlier count (within 2m of median)
        residuals = np.linalg.norm(deltas - delta_xy, axis=1)
        n_inliers = int(np.sum(residuals < 2.0))
        
        if n_inliers < self.min_matches:
            return None
        
        # Refine with inliers only
        inlier_mask = residuals < 2.0
        delta_xy = np.mean(deltas[inlier_mask], axis=0)
        
        return delta_xy[0], delta_xy[1], n_inliers
    
    def add_frame(self, gray, pose, K):
        """
        Add a new frame to the pose graph.
        
        Args:
            gray: Grayscale image (for SIFT)
            pose: 4x4 pose matrix (from metadata)
            K: 3x3 camera intrinsics
            
        Returns:
            corrected_pose: The pose matrix with translation correction applied
        """
        idx = len(self.frames)
        gps_xy = pose[:3, 3][:2].copy()  # NED X, Y from GPS
        corners = self._compute_ground_corners(pose, K)
        
        # Detect features
        kp, des = self.sift.detectAndCompute(gray, None)
        
        self.frames.append({
            'gray': None,  # Don't store images to save memory
            'pose': pose.copy(),
            'gps_xy': gps_xy,
            'corners': corners,
            'keypoints': kp,
            'descriptors': des,
        })
        self.corrections.append(np.zeros(2))  # Initial correction = 0
        
        # Find overlapping previous frames and match
        n_new_edges = 0
        # Check recent frames (sliding window) + any with IoU overlap
        candidates = []
        for j in range(max(0, idx - 10), idx):  # Last 10 frames
            candidates.append(j)
        
        # Also check by IoU for non-sequential overlaps
        for j in range(max(0, idx - 10)):
            if corners is not None and self.frames[j]['corners'] is not None:
                iou = self._compute_iou(corners, self.frames[j]['corners'])
                if iou > self.iou_threshold:
                    candidates.append(j)
        
        candidates = list(set(candidates))
        
        for j in candidates:
            result = self._match_pair(j, idx, K)
            if result is not None:
                dx, dy, n_inliers = result
                self.edges.append((j, idx, np.array([dx, dy])))
                n_new_edges += 1
                logger.info(f"PoseGraph: Edge ({j},{idx}): delta=({dx:.2f},{dy:.2f})m, {n_inliers} inliers")
        
        # Re-solve if we have new edges
        if n_new_edges > 0:
            self._solve()
            self._dirty = True
        
        # Return corrected pose
        return self.get_corrected_pose(idx)
    
    def _solve(self):
        """
        Solve the sparse linear system for optimal translations.
        
        Variables: [dx_0, dy_0, dx_1, dy_1, ..., dx_n, dy_n] (2 per frame)
        
        Equations:
        - GPS prior for each frame i:   w_gps * (t_i + correction_i) = w_gps * gps_i
          => w_gps * correction_i = w_gps * 0  (correction is relative to GPS)
          
        - Pairwise constraint (i, j):  w_match * ((t_j + corr_j) - (t_i + corr_i)) = w_match * (t_j - t_i + delta_ij)
          => w_match * (corr_j - corr_i) = w_match * delta_ij
          where delta_ij is the measured relative translation from features.
          
          But delta_ij measures how the ground points from i and j disagree.
          More precisely: delta_ij = (expected_offset) - (actual_offset_from_GPS)
          So: w_match * (corr_j - corr_i) = w_match * (delta_ij - (gps_j - gps_i))
          Wait — let me re-derive.
          
          From feature matching, pts_j - pts_i gives how much j's ground projection
          differs from i's for the same 3D point. If GPS were perfect, this should be 0.
          So delta_ij IS the error. We want:
            (gps_j + corr_j) - (gps_i + corr_i) = (gps_j - gps_i) - delta_ij
          => corr_j - corr_i = -delta_ij
        """
        n = len(self.frames)
        if n == 0:
            return
        
        n_gps = n           # One GPS prior per frame
        n_edges = len(self.edges)
        n_rows = (n_gps + n_edges) * 2  # x2 for X and Y
        n_cols = n * 2
        
        A = lil_matrix((n_rows, n_cols), dtype=np.float64)
        b = np.zeros(n_rows, dtype=np.float64)
        
        row = 0
        
        # GPS priors: correction_i should be 0 (stay near GPS)
        for i in range(n):
            # X component
            A[row, 2*i] = self.w_gps
            b[row] = 0.0  # correction = 0 means stay at GPS
            row += 1
            # Y component
            A[row, 2*i + 1] = self.w_gps
            b[row] = 0.0
            row += 1
        
        # Pairwise constraints: corr_j - corr_i = -delta_ij
        for (i, j, delta_ij) in self.edges:
            # X component
            A[row, 2*j] = self.w_match
            A[row, 2*i] = -self.w_match
            b[row] = self.w_match * (-delta_ij[0])
            row += 1
            # Y component
            A[row, 2*j + 1] = self.w_match
            A[row, 2*i + 1] = -self.w_match
            b[row] = self.w_match * (-delta_ij[1])
            row += 1
        
        # Solve via LSQR (works great for sparse least-squares)
        A_csr = A.tocsr()
        result_x = lsqr(A_csr, b)
        x = result_x[0]
        
        # Extract corrections
        for i in range(n):
            self.corrections[i] = np.array([x[2*i], x[2*i + 1]])
        
        # Log stats
        max_corr = max(np.linalg.norm(c) for c in self.corrections) if n > 0 else 0
        logger.info(f"PoseGraph solved: {n} frames, {n_edges} edges, max correction={max_corr:.3f}m")
        print(f"PoseGraph solved: {n} frames, {n_edges} edges, max correction={max_corr:.3f}m")
    
    def get_corrected_pose(self, idx):
        """Return pose matrix with translation correction applied."""
        pose = self.frames[idx]['pose'].copy()
        corr = self.corrections[idx]
        pose[0, 3] += corr[0]  # NED X (North)
        pose[1, 3] += corr[1]  # NED Y (East)
        return pose
    
    def get_all_corrections(self):
        """Return list of (idx, correction_magnitude) for all frames."""
        return [(i, np.linalg.norm(c)) for i, c in enumerate(self.corrections)]
    
    def get_stats(self):
        """Return optimizer statistics for display."""
        n = len(self.frames)
        if n == 0:
            return {'n_frames': 0, 'n_edges': 0, 'max_correction': 0.0, 'avg_correction': 0.0}
        magnitudes = [np.linalg.norm(c) for c in self.corrections]
        return {
            'n_frames': n,
            'n_edges': len(self.edges),
            'max_correction': max(magnitudes),
            'avg_correction': np.mean(magnitudes),
        }
    
    def reset(self):
        """Clear all accumulated data."""
        self.frames.clear()
        self.edges.clear()
        self.corrections.clear()
        self._dirty = False


# ================= MAPPER ENGINE =================
class MultiBandMap2D:
    def __init__(self, resolution=0.05, band_num=2, tile_size=512):
        """
        Multi-band blending mapper for aerial imagery.
        
        Args:
            resolution: meters per pixel in output map
            band_num: number of pyramid levels for blending
            tile_size: size of each map tile in pixels
        """
        self.resolution = resolution
        self.band_num = band_num
        self.tile_size = tile_size
        self.tiles = {}
        self.weight_mask = None
        self.last_frame_shape = None
        self.lock = threading.Lock()
        self.paused = False
        self.last_pts_metric = None # For IoU calculation
        
        # Flight path tracking: list of (east_metric, north_metric) camera positions
        self.flight_path = []  # [(map_x, map_y), ...] in metric display coords
        self.frame_count = 0

    def calculate_gsd(self, altitude, focal_len_mm, sensor_width_mm, img_width_px):
        """
        Calculate Ground Sample Distance (meters per pixel).
        GSD = (SensorWidth * Altitude) / (FocalLength * ImageWidth)
        """
        if focal_len_mm == 0 or img_width_px == 0:
            return 0
        return (sensor_width_mm * altitude) / (focal_len_mm * img_width_px)

    def calculate_iou(self, pts1, pts2):
        """
        Calculate Intersection over Union for two sets of 4 points on the ground plane.
        Uses a mask-based approach for robustness with arbitrary quadrilaterals.
        """
        if pts1 is None or pts2 is None:
            return 0.0
            
        try:
            # Determine bounding box for both
            all_pts = np.vstack([pts1, pts2])
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            
            # Map coordinates to a pixel grid (e.g., 0.1m resolution for calculation)
            calc_res = 0.1
            width = int((xmax - xmin) / calc_res) + 2
            height = int((ymax - ymin) / calc_res) + 2
            
            # Create masks
            mask1 = np.zeros((height, width), dtype=np.uint8)
            mask2 = np.zeros((height, width), dtype=np.uint8)
            
            # Convert pts to local pixel coords
            p1 = ((pts1 - [xmin, ymin]) / calc_res).astype(np.int32)
            p2 = ((pts2 - [xmin, ymin]) / calc_res).astype(np.int32)
            
            # Fill polygons
            cv2.fillPoly(mask1, [p1], 255)
            cv2.fillPoly(mask2, [p2], 255)
            
            # Intersection and Union
            inter = cv2.bitwise_and(mask1, mask2)
            union = cv2.bitwise_or(mask1, mask2)
            
            inter_area = cv2.countNonZero(inter)
            union_area = cv2.countNonZero(union)
            
            if union_area == 0:
                return 0.0
                
            return inter_area / union_area
        except Exception as e:
            print(f"IoU Error: {e}")
            return 0.0

    def _get_weight_mask(self, shape):
        """Create Gaussian-like weight mask favoring image center."""
        if self.weight_mask is not None and self.last_frame_shape == shape:
            return self.weight_mask
        h, w = shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        maxd = np.sqrt(center_x**2 + center_y**2)
        mask = np.clip(1.0 - dist / maxd, 1e-5, 1.0).astype(np.float32)
        self.weight_mask = mask * mask  # Squared for stronger center bias
        self.last_frame_shape = shape
        return self.weight_mask

    def _create_laplace_pyr(self, img):
        """Create Laplacian pyramid for multi-band blending."""
        pyr, cur = [], img
        for _ in range(self.band_num):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cv2.subtract(cur, up))
            cur = down
        pyr.append(cur)
        return pyr

    def feed(self, frame, pose_matrix, camera_matrix, plane_height=0.0):
        """
        Add a new frame to the map.
        
        Args:
            frame: Input image (BGR)
            pose_matrix: 4x4 camera pose in NED frame (Z points down)
            camera_matrix: 3x3 camera intrinsic matrix
            plane_height: Z-coordinate of ground plane (0 for ground level)
        
        Note: In NED frame, altitude should be NEGATIVE (below origin).
        """
        if self.paused: 
            return False
            
        h, w = frame.shape[:2]

        R = pose_matrix[:3, :3]  # Rotation matrix
        t = pose_matrix[:3, 3]   # Translation vector (camera position)
        
        # Image corner points
        pts_src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        
        # --- PROJECT IMAGE CORNERS TO GROUND PLANE ---
        # Camera coordinates from image coordinates
        u = np.array([0, w-1, w-1, 0], np.float32)
        v = np.array([0, 0, h-1, h-1], np.float32)
        
        # Unproject to normalized camera coordinates
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        # Ray directions in camera frame
        # Standard CV camera: +X right, +Y down, +Z forward
        # After pitch adjustment, +Z points down toward ground
        ray_camera = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=1)
        ray_camera /= np.linalg.norm(ray_camera, axis=1, keepdims=True)
        
        # Transform rays to world frame (NED: X=North, Y=East, Z=Down)
        ray_world = (R @ ray_camera.T).T
        
        # In NED frame with camera at negative altitude:
        # - Ground plane is at Z = 0
        # - Camera is at Z = -altitude (negative)
        # - Rays should have positive Z component (pointing toward larger Z = down)
        # Intersect: P = camera_pos + λ * ray where P_z = plane_height
        lams = (plane_height - t[2]) / ray_world[:, 2]
        
        # Check for rays pointing away from ground
        if np.any(lams < 0):
            print("Warning: Some rays don't intersect ground plane")
            return False
        
        # World coordinates of corner points
        pts_world = t + lams[:, None] * ray_world
        
        # Map world NED (X=N, Y=E, Z=D) to Map Pixels (X=East, Y=-North)
        # This keeps North UP and East RIGHT.
        pts_metric = np.zeros((4, 2), np.float32)
        pts_metric[:, 0] = pts_world[:, 1]  # Easting
        pts_metric[:, 1] = -pts_world[:, 0] # -Northing
        
        # Track camera position for flight path overlay
        cam_map_x = t[1]   # Easting
        cam_map_y = -t[0]  # -Northing
        self.flight_path.append((cam_map_x, cam_map_y))
        self.frame_count += 1
        
        # Calculate coverage area
        xmin, xmax = pts_metric[:, 0].min(), pts_metric[:, 0].max()
        ymin, ymax = pts_metric[:, 1].min(), pts_metric[:, 1].max()
        
        # Let's use real labels for the print
        e_min, e_max = pts_world[:, 1].min(), pts_world[:, 1].max()
        n_min, n_max = pts_world[:, 0].min(), pts_world[:, 0].max()
        
        print(f"\nFrame coverage: {e_max-e_min:.2f}m (E-W) x {n_max-n_min:.2f}m (N-S) at altitude {-t[2]:.2f}m")
        
        # Calculate and Log Metrics
        # Altitude in NED is negative, so use -t[2]
        alt = -t[2]
        gsd = self.calculate_gsd(alt, camera_matrix[0,0], 1.0, 1.0) # Approx if fx is passed directly
        # If we have focal_len/sensor_w we can be more exact, but resolution is usually fixed.
        
        iou = self.calculate_iou(self.last_pts_metric, pts_metric)
        self.last_pts_metric = pts_metric.copy()
        
        print(f"Metrics: GSD={gsd:.4f}m/px | IoU with prev={iou:.2%}")
        
        # Convert to pixel coordinates in map
        pts_pixels = pts_metric / self.resolution
        # Use raw projected points directly (TL, TR, BR, BL order)
        pts_pixels = pts_pixels.astype(np.float32)
        
        # Shift to local coordinate system for warping
        local_offset = pts_pixels.min(axis=0)
        pts_pixels_local = pts_pixels - local_offset
        
        # Calculate output size with padding
        out_w = int(np.ceil(pts_pixels_local[:, 0].max())) + 10
        out_h = int(np.ceil(pts_pixels_local[:, 1].max())) + 10
        
        # Warp image to map coordinates
        H = cv2.getPerspectiveTransform(pts_src, pts_pixels_local)
        warped = cv2.warpPerspective(frame, H, (out_w, out_h))
        wmask = cv2.warpPerspective(self._get_weight_mask(frame.shape), H, (out_w, out_h))
        
        # Create pyramids for blending
        pyr_img = self._create_laplace_pyr(warped.astype(np.float32))
        pyr_w = [wmask]
        for _ in range(self.band_num):
            pyr_w.append(cv2.pyrDown(pyr_w[-1]))
        
        # Determine which tiles this frame touches
        def tile_index(v):
            return int(np.floor(v / (self.resolution * self.tile_size)))
        
        tminx, tmaxx = tile_index(xmin), tile_index(xmax)
        tminy, tmaxy = tile_index(ymin), tile_index(ymax)
        
        # Update tiles
        with self.lock:
            for tx in range(tminx, tmaxx + 1):
                for ty in range(tminy, tmaxy + 1):
                    key = (tx, ty)
                    
                    # Initialize tile if needed
                    if key not in self.tiles:
                        self.tiles[key] = {
                            'pyr': [np.zeros((self.tile_size // (2**i), self.tile_size // (2**i), 3), 
                                           np.float32) for i in range(self.band_num + 1)],
                            'w': [np.zeros((self.tile_size // (2**i), self.tile_size // (2**i)), 
                                         np.float32) for i in range(self.band_num + 1)]
                        }
                    
                    # Calculate tile position in world coordinates
                    tile_x_world = tx * self.tile_size * self.resolution
                    tile_y_world = ty * self.tile_size * self.resolution
                    
                    # Offset from world to warped image coordinates
                    sx = int((tile_x_world - xmin) / self.resolution)
                    sy = int((tile_y_world - ymin) / self.resolution)
                    
                    # Blend at each pyramid level
                    for i in range(self.band_num + 1):
                        scale = 2 ** i
                        ts = self.tile_size // scale
                        lx, ly = sx // scale, sy // scale
                        
                        # Calculate valid region in warped image
                        x0 = max(0, lx)
                        y0 = max(0, ly)
                        x1 = min(lx + ts, pyr_img[i].shape[1])
                        y1 = min(ly + ts, pyr_img[i].shape[0])
                        
                        w0, h0 = x1 - x0, y1 - y0
                        if w0 <= 0 or h0 <= 0:
                            continue
                        
                        # Calculate position in tile
                        dx, dy = x0 - lx, y0 - ly
                        
                        # Extract regions
                        img_patch = pyr_img[i][y0:y1, x0:x1]
                        wt_patch = pyr_w[i][y0:y1, x0:x1]
                        
                        tile_img = self.tiles[key]['pyr'][i][dy:dy+h0, dx:dx+w0]
                        tile_w = self.tiles[key]['w'][i][dy:dy+h0, dx:dx+w0]
                        
                        # Update where new weight is higher
                        mask = wt_patch > tile_w
                        tile_w[mask] = wt_patch[mask]
                        mask_3d = mask.repeat(3).reshape(mask.shape + (3,))
                        tile_img[mask_3d] = img_patch[mask_3d]
        
        return True

    def render_map(self, quality_lvl=0, show_flight_path=False):
        """
        Render the complete map by reconstructing from pyramids.
        
        Args:
            quality_lvl: pyramid level to render (0 = full quality)
            show_flight_path: if True, draw flight path overlay on the map
        """
        with self.lock:
            if not self.tiles:
                return None
            
            ks = list(self.tiles.keys())
            minx = min(k[0] for k in ks)
            maxx = max(k[0] for k in ks)
            miny = min(k[1] for k in ks)
            maxy = max(k[1] for k in ks)
            
            ts = self.tile_size // (2 ** quality_lvl)
            canvas = np.zeros(((maxy - miny + 1) * ts, (maxx - minx + 1) * ts, 3), np.uint8)
            
            for (tx, ty), d in self.tiles.items():
                # Reconstruct from pyramid
                cur = d['pyr'][-1]
                for i in range(self.band_num - 1, quality_lvl - 1, -1):
                    cur = cv2.add(d['pyr'][i], 
                                 cv2.pyrUp(cur, dstsize=(d['pyr'][i].shape[1], 
                                                        d['pyr'][i].shape[0])))
                
                # Place in canvas
                ox = (tx - minx) * ts
                oy = (ty - miny) * ts
                canvas[oy:oy + cur.shape[0], ox:ox + cur.shape[1]] = np.clip(cur, 0, 255)
        
            # Draw flight path overlay
            if show_flight_path and len(self.flight_path) >= 2:
                scale = 2 ** quality_lvl
                # Convert metric positions to canvas pixel positions
                # Tile origin in metric space
                origin_x_metric = minx * self.tile_size * self.resolution
                origin_y_metric = miny * self.tile_size * self.resolution
                
                path_pts = []
                for (mx, my) in self.flight_path:
                    px = int((mx - origin_x_metric) / self.resolution / scale)
                    py = int((my - origin_y_metric) / self.resolution / scale)
                    path_pts.append((px, py))
                
                # Draw path lines
                for i in range(1, len(path_pts)):
                    cv2.line(canvas, path_pts[i-1], path_pts[i], (0, 200, 255), 2, cv2.LINE_AA)
                
                # Draw waypoints
                for i, pt in enumerate(path_pts):
                    # Start = green, end = red, middle = cyan
                    if i == 0:
                        color = (0, 255, 0)
                        radius = 8
                    elif i == len(path_pts) - 1:
                        color = (0, 0, 255)
                        radius = 8
                    else:
                        color = (255, 255, 0)
                        radius = 4
                    cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, radius, (0, 0, 0), 1, cv2.LINE_AA)  # Black border
                
                # Label start and end
                if path_pts:
                    cv2.putText(canvas, "S", (path_pts[0][0]+10, path_pts[0][1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, "E", (path_pts[-1][0]+10, path_pts[-1][1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
        return canvas
    
    def get_memory_usage(self):
        """Estimate memory usage of tile data in bytes."""
        total = 0
        with self.lock:
            for key, tile_data in self.tiles.items():
                for arr in tile_data['pyr']:
                    total += arr.nbytes
                for arr in tile_data['w']:
                    total += arr.nbytes
        return total
