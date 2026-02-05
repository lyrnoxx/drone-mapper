import cv2
import numpy as np
from .core import MultiBandMap2D, PoseExtractor

class MapperService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MapperService()
        return cls._instance

    def __init__(self):
        # DJI M3T camera parameters (approximate)
        self.update_parameters(24.0, 36.0, 4000)
        
        # Initialize engine and pose extractor
        self.mapper = MultiBandMap2D(resolution=0.5, band_num=2, tile_size=512)
        self.pose_extractor = PoseExtractor()

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
        
    def process_image(self, img_bytes):
        """Process a raw image: extract pose -> feed to mapper."""
        # Extract pose
        pose_matrix = self.pose_extractor.extract_pose(img_bytes)
        if pose_matrix is None:
            return {"status": "error", "message": "Could not extract pose from EXIF"}
            
        # Decode and rotate image
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
        if frame is None:
             return {"status": "error", "message": "Failed to decode image"}
             
        # DJI images are processed in their raw orientation (landscape)
        # Pose extraction accounts for this orientation.
        
        # Connect mapper to pose_extractor for satellite reference math
        self.mapper.pose_extractor_ref = self.pose_extractor
        
        # Feed to mapper with metadata for satellite alignment
        success = self.mapper.feed(frame, pose_matrix, self.K, metadata=self.pose_extractor.last_metadata)
        
        if success:
             return {"status": "success", "message": "Image processed"}
        else:
             return {"status": "ignored", "message": "Image ignored (paused or invalid rays)"}

    def get_map_image(self):
        """Return the current map as an encoded JPEG buffer."""
        img = self.mapper.render_map(0)
        if img is None:
            return None
        
        # Convert to BGR -> RGB for web usage if we were sending raw pixels,
        # but for JPEG encoding cv2 uses BGR default, so it's fine.
        # Actually standard web images are RGB, but if we encode to JPEG via imencode, it expects BGR.
        # So providing the BGR image from render_map directly to imencode is correct.
        
        success, buffer = cv2.imencode('.jpg', img)
        if not success:
            return None
        return buffer.tobytes()

    def reset_map(self):
        self.mapper = MultiBandMap2D(resolution=0.5, band_num=2, tile_size=512)
        return True
