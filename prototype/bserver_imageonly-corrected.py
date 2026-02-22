import cv2
import numpy as np
import socket
import struct
import threading
import time
import re
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
            
            return T
            
        except Exception as e:
            print(f"Error extracting pose: {e}")
            return None


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
        self.last_center_px = None # Tracks latest frame position in map pixels

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
        
        # Calculate coverage area
        xmin, xmax = pts_metric[:, 0].min(), pts_metric[:, 0].max()
        ymin, ymax = pts_metric[:, 1].min(), pts_metric[:, 1].max()
        
        print(f"\nFrame coverage: {xmax-xmin:.2f}m (E-W) x {ymax-ymin:.2f}m (N-S) at altitude {-t[2]:.2f}m")
        
        # Convert to pixel coordinates in map
        # Use simple order (TL, TR, BR, BL as generated from pts_src unprojection)
        pts_pixels = (pts_metric / self.resolution).astype(np.float32)
        
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
        
        # Update tiles
        # (tiles update logic)
        
        # Track latest center for visualization
        self.last_center_px = pts_pixels.mean(axis=0)
        
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

# ================= NETWORK UTILITIES =================
def recv_all(sock, n):
    """Receive exactly n bytes from socket."""
    data = b''
    while len(data) < n:
        p = sock.recv(n - len(data))
        if not p:
            return None
        data += p
    return data

def handle_conn(conn, mapper, K, pose_extractor):
    """Handle incoming connection and process frames (image-only protocol)."""
    print("Client connected")
    frame_count = 0
    
    while True:
        # Read message type
        t = recv_all(conn, 1)
        if not t or t == b'S':
            break
        
        if t != b'I':
            print(f"Unknown message type: {t}")
            continue
        
        # Read image length
        l = recv_all(conn, 4)
        if not l:
            break
        
        L = struct.unpack('<I', l)[0]
        
        # Read image data
        img_bytes = recv_all(conn, L)
        if not img_bytes:
            break
        
        # Extract pose from EXIF
        pose_matrix = pose_extractor.extract_pose(img_bytes)
        if pose_matrix is None:
            print(f"Skipping frame {frame_count}: Could not extract pose")
            frame_count += 1
            continue
        
        # Decode image
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
        # REMOVED: cv2.rotate. EXIF metadata is for the raw landscape orientation.
        
        # Feed to mapper
        mapper.feed(frame, pose_matrix, K)
        frame_count += 1
    
    conn.close()
    print(f"Client disconnected. Processed {frame_count} frames.")

# ================= SERVER =================
if __name__ == "__main__":
    # DJI M3T camera parameters (corrected for raw 4000x3000 capture)
    # W=4000, H=3000 -> cx=2000, cy=1500
    K = np.array([
        [2666, 0, 2000],
        [0, 2666, 1500],
        [0, 0, 1]
    ])
    
    print("Camera Intrinsics:")
    print(K)
    
    # Create mapper with 10cm resolution (Improved from 50cm)
    mapper = MultiBandMap2D(resolution=0.1, band_num=2, tile_size=512)
    
    # Create pose extractor
    pose_extractor = PoseExtractor()
    
    # Start server
    s = socket.socket()
    s.bind(("0.0.0.0", 5006))
    s.listen(5)
    print("\nImage-Only Server listening on port 5006...")
    print("Controls: [P] Pause/Play | [S] Save Image | [ESC] Exit")
    
    # Accept connections in background
    def accept_loop():
        while True:
            conn, addr = s.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=handle_conn, args=(conn, mapper, K, pose_extractor), daemon=True).start()
    
    threading.Thread(target=accept_loop, daemon=True).start()
    
    # Display loop
    cv2.namedWindow("Live Map", cv2.WINDOW_NORMAL)
    
    # Viewport state
    view_cx, view_cy = 0, 0 
    zoom_val = 1.0
    auto_follow = False # Set to False to stop jumping between drones
    initialized_view = False
    VIEW_W, VIEW_H = 1280, 720
    
    print("\nDisplay Controls:")
    print(" [WASD]  Pan View")
    print(" [Q/E]   Zoom Out/In")
    print(" [R]     Resume Auto-Follow (Once-off sync then stays)")
    print(" [P]     Pause Engine")
    print(" [S]     Save Map Snapshot")
    
    while True:
        img = mapper.render_map(0)
        
        if img is not None:
            h, w = img.shape[:2]
            ks = list(mapper.tiles.keys())
            if ks:
                minx_t = min(k[0] for k in ks)
                miny_t = min(k[1] for k in ks)
                origin_px_x = minx_t * mapper.tile_size
                origin_px_y = miny_t * mapper.tile_size
                
                # Initialize or follow if requested
                if mapper.last_center_px is not None:
                    if not initialized_view or auto_follow:
                        view_cx = int(mapper.last_center_px[0]) - origin_px_x
                        view_cy = int(mapper.last_center_px[1]) - origin_px_y
                        initialized_view = True
                        if auto_follow:
                             auto_follow = False # Stop following after one sync
                             print("Synced to latest frame")

                # Calculate crop with zoom
                cur_w = int(VIEW_W / zoom_val)
                cur_h = int(VIEW_H / zoom_val)
                
                # Desired viewport in map-origin-relative pixels (can be negative)
                vx0 = view_cx - cur_w // 2
                vy0 = view_cy - cur_h // 2
                vx1 = vx0 + cur_w
                vy1 = vy0 + cur_h
                
                # Intersection with current engine image
                ix0, ix1 = max(0, vx0), min(w, vx1)
                iy0, iy1 = max(0, vy0), min(h, vy1)
                
                # Create padded black canvas for viewport
                display_bg = np.zeros((cur_h, cur_w, 3), np.uint8)
                
                # Copy valid map part if there is an intersection
                if ix1 > ix0 and iy1 > iy0:
                    # Destination coordinates in the black canvas
                    dx0, dy0 = ix0 - vx0, iy0 - vy0
                    dx1, dy1 = dx0 + (ix1 - ix0), dy0 + (iy1 - iy0)
                    
                    display_bg[dy0:dy1, dx0:dx1] = img[iy0:iy1, ix0:ix1]
                
                # Resize specifically to window size (INTER_CUBIC for higher quality)
                display_img = cv2.resize(display_bg, (VIEW_W, VIEW_H), interpolation=cv2.INTER_CUBIC)
            else:
                display_img = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)
                cv2.putText(display_img, "Waiting for Data...", (VIEW_W//2-150, VIEW_H//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if mapper.paused:
                cv2.putText(display_img, "PAUSED", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            cv2.imshow("Live Map", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('w'): 
            view_cy -= int(50 / zoom_val); auto_follow = False
        elif key == ord('s'): 
            view_cy += int(50 / zoom_val); auto_follow = False
        elif key == ord('a'): 
            view_cx -= int(50 / zoom_val); auto_follow = False
        elif key == ord('d'): 
            view_cx += int(50 / zoom_val); auto_follow = False
        elif key == ord('q'): 
            zoom_val *= 0.8; auto_follow = False # Zoom Out
        elif key == ord('e'): 
            zoom_val *= 1.25; auto_follow = False # Zoom In
        elif key == ord('r'): 
            zoom_val = 1.0; auto_follow = True; print("Auto-Follow Resumed")
        elif key == ord('p') or key == ord('P'):
            mapper.paused = not mapper.paused
            print(f"Mapper Status: {'Paused' if mapper.paused else 'Playing'}")
        elif key == ord('s') or key == ord('S'):
            if img is not None:
                fname = f"map_{int(time.time())}.png"
                cv2.imwrite(fname, img)
                print(f"Saved: {fname}")
    
    cv2.destroyAllWindows()
