import cv2
import numpy as np
import threading
import re
import math
import requests
import io
import os
from pyproj import Proj
from scipy.spatial.transform import Rotation as R

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


# ================= SATELLITE REGISTRATION =================
class SatelliteManager:
    """Fetches and manages geo-referenced satellite tiles for ground truth alignment."""
    def __init__(self, zoom=19):
        self.zoom = zoom
        self.tiles = {} # (x, y) -> image
        self.lock = threading.Lock()
        
    def _get_tile_indices(self, lat, lon):
        n = 2.0 ** self.zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return x, y

    def _get_tile_extent(self, x, y):
        """Calculate the lat/lon extent of a tile."""
        n = 2.0 ** self.zoom
        lon_left = x / n * 360.0 - 180.0
        lon_right = (x + 1) / n * 360.0 - 180.0
        
        def y_to_lat(y_idx):
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_idx / n)))
            return math.degrees(lat_rad)
            
        lat_top = y_to_lat(y)
        lat_bottom = y_to_lat(y + 1)
        
        return (lat_bottom, lat_top), (lon_left, lon_right)

    def get_reference_image(self, lat, lon, proj, origin_x, origin_y):
        """
        Fetch a 3x3 grid of satellite tiles and stitch them.
        """
        center_tx, center_ty = self._get_tile_indices(lat, lon)
        
        # Grid range
        tiles_list = []
        for dy in [-1, 0, 1]:
            row = []
            for dx in [-1, 0, 1]:
                tx, ty = center_tx + dx, center_ty + dy
                key = (tx, ty)
                
                with self.lock:
                    if key not in self.tiles:
                        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{ty}/{tx}"
                        try:
                            response = requests.get(url, timeout=5)
                            if response.status_code == 200:
                                img_arr = np.frombuffer(response.content, np.uint8)
                                self.tiles[key] = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                                print(f"Downloaded Satellite Tile {key}")
                            else:
                                self.tiles[key] = np.zeros((256, 256, 3), np.uint8) # Placeholder
                        except Exception:
                            self.tiles[key] = np.zeros((256, 256, 3), np.uint8)
                    
                    row.append(self.tiles[key])
            tiles_list.append(row)
            
        # Stitch
        stitched = cv2.vconcat([cv2.hconcat(r) for r in tiles_list])
        
        # Get extent of the grid
        # Leftmost lon, bottommost lat (min) to Rightmost lon, topmost lat (max)
        _, (lon_min, _) = self._get_tile_extent(center_tx - 1, center_ty)
        _, (_, lon_max) = self._get_tile_extent(center_tx + 1, center_ty)
        (lat_min, _), _ = self._get_tile_extent(center_tx, center_ty + 1)
        (_, lat_max), _ = self._get_tile_extent(center_tx, center_ty - 1)
        
        # Project corners
        mx_min, my_min = proj(lon_min, lat_min)
        mx_max, my_max = proj(lon_max, lat_max)
        
        lx_min, ly_min = mx_min - origin_x, my_min - origin_y
        lx_max, ly_max = mx_max - origin_x, my_max - origin_y
        
        return stitched, [lx_min, ly_min, lx_max, ly_max]

class PoseRefiner:
    """Aligns drone imagery to satellite baseline using feature matching."""
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        
    def refine(self, frame, initial_pose, sat_img, sat_extent, K):
        """
        initial_pose: 4x4 rough matrix (drone's initial estimate)
        sat_img: Satellite ground truth image
        sat_extent: [xmin, ymin, xmax, ymax] in metric
        K: Camera intrinsics
        """
        if sat_img is None:
            return initial_pose, 0.0
            
        # 1. Feature Matching (as before)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_sat = cv2.cvtColor(sat_img, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = self.sift.detectAndCompute(gray_frame, None)
        kp_s, des_s = self.sift.detectAndCompute(gray_sat, None)
        
        if des_f is None or des_s is None:
            return initial_pose, 0.0
            
        matches = self.flann.knnMatch(des_f, des_s, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        
        if len(good) < 20:
            return initial_pose, 0.0
            
        pts_f_px = np.float32([kp_f[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_s_px = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # 2. Convert Drone pixels to Rough Ground Coordinates (Metric)
        # Use simple unprojection to ground plane at initial_pose
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u, v = pts_f_px[:, 0, 0], pts_f_px[:, 0, 1]
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        ray_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=1)
        ray_cam /= np.linalg.norm(ray_cam, axis=1, keepdims=True)
        
        R_initial = initial_pose[:3, :3]
        T_initial = initial_pose[:3, 3]
        ray_world = (R_initial @ ray_cam.T).T
        
        # Intersection with plane z=0 (NED frame, camera is at T_initial[2] which is negative altitude)
        lams = -T_initial[2] / ray_world[:, 2]
        pts_drone_metric = T_initial + lams[:, None] * ray_world
        pts_drone_metric = pts_drone_metric[:, :2] # X, Y only
        
        # 3. Convert Satellite pixels to Ground Coordinates (Metric)
        sx_min, sy_min, sx_max, sy_max = sat_extent
        sw, sh = gray_sat.shape[1], gray_sat.shape[0]
        pts_sat_metric = np.zeros_like(pts_drone_metric)
        pts_sat_metric[:, 0] = sx_min + (pts_s_px[:, 0, 0] / sw) * (sx_max - sx_min)
        pts_sat_metric[:, 1] = sy_max - (pts_s_px[:, 0, 1] / sh) * (sy_max - sy_min) # Y inverted
        
        try:
            # 4. Center coordinates around Drone to avoid Origin-Rotation issues
            centroid_drone = np.mean(pts_drone_metric, axis=0)
            A_centered = pts_drone_metric - centroid_drone
            B_centered = pts_sat_metric - centroid_drone
            
            # 5. Solve for 2D Rigid/Similarity Transform using RANSAC
            # estimateAffinePartial2D finds Translation + Rotation + (Uniform) Scale
            M, inliers = cv2.estimateAffinePartial2D(A_centered, B_centered, method=cv2.RANSAC)
            
            if M is None or np.sum(inliers) < 10:
                return initial_pose, 0.0
            
            # M is [[cos, -sin, tx], [sin, cos, ty]]
            R_2d = M[:, 0:2]
            t_2d = M[:, 2] # This is the correction RELATIVE to the drone's centroid
            
            # Safety Check: Discard extreme jumps (> 100m)
            dist_err = np.linalg.norm(t_2d)
            if dist_err > 100.0:
                print(f"Refinement Ignored: Extreme offset detected ({dist_err:.1f}m)")
                return initial_pose, 0.0
            
            # 6. Apply correction LOCALLY
            # New Pose = Translate Correction * Rotate Correction (at drone) * Old Pose
            # But simpler: Update t and R directly 
            
            # Translation update
            refined_pose = initial_pose.copy()
            refined_pose[0, 3] += t_2d[0]
            refined_pose[1, 3] += t_2d[1]
            
            # Rotation update (Compass/Yaw fix)
            # Apply R_2d only to the XY components of the rotation
            R_old = initial_pose[:3, :3]
            # Convert R_2d to 3x3
            R_3d = np.eye(3)
            R_3d[0:2, 0:2] = R_2d
            
            refined_pose[:3, :3] = R_3d @ R_old
            
            print(f"Refinement Success: {np.sum(inliers)} inliers. Offset: {dist_err:.2f}m")
            return refined_pose, float(np.sum(inliers))
            
        except Exception as e:
            print(f"Refinement Solver Error: {e}")
            return initial_pose, 0.0


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
        
        # Satellite Registration
        self.sat_manager = SatelliteManager(zoom=18)
        self.pose_refiner = PoseRefiner()
        self.enable_refinement = False
        self.pose_extractor_ref = None # Set this to the active PoseExtractor

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

    def feed(self, frame, pose_matrix, camera_matrix, plane_height=0.0, metadata=None):
        """
        Add a new frame to the map.
        
        Args:
            frame: Input image (BGR)
            pose_matrix: 4x4 camera pose in NED frame (Z points down)
            camera_matrix: 3x3 camera intrinsic matrix
            plane_height: Z-coordinate of ground plane (0 for ground level)
            metadata: DJI XMP metadata for satellite registration
        
        Note: In NED frame, altitude should be NEGATIVE (below origin).
        """
        if self.paused: 
            return False
            
        h, w = frame.shape[:2]
        
        # Satellite Refinement
        if self.enable_refinement and metadata is not None:
            lat = metadata.get('GPSLatitude')
            lon = metadata.get('GPSLongitude')
            if lat and lon:
                # Fetch reference tile
                sat_img, sat_extent = self.sat_manager.get_reference_image(
                    lat, lon, self.pose_extractor_ref.proj, 
                    self.pose_extractor_ref.origin_x, self.pose_extractor_ref.origin_y
                )
                # Refine pose
                pose_matrix, _ = self.pose_refiner.refine(frame, pose_matrix, sat_img, sat_extent, camera_matrix)

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

    def render_map(self, quality_lvl=0):
        """
        Render the complete map by reconstructing from pyramids.
        
        Args:
            quality_lvl: pyramid level to render (0 = full quality)
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
        
        return canvas
