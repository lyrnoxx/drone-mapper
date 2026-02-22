import os
import cv2
import numpy as np
import sys

# Ensure backend can be imported
sys.path.append(os.getcwd())

from backend.core import MultiBandMap2D, PoseExtractor

def analyze():
    img_folder = "images-true"
    if not os.path.exists(img_folder):
        print(f"Error: {img_folder} not found")
        return

    # DJI M3T camera parameters (approximate)
    # focal_len_mm = 4.4, sensor_width_mm = 6.17, img_width_px = 4000
    # Actually, your code uses: 
    # fx = img_width_px * (focal_len_mm / sensor_width_mm)
    # Let's use the ones from bserver_imageonly.py
    K = np.array([
        [2666, 0, 2000],
        [0, 2666, 1500],
        [0, 0, 1]
    ])

    mapper = MultiBandMap2D(resolution=0.5, band_num=2, tile_size=512)
    extractor = PoseExtractor()

    images = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg'))])
    
    print(f"{'Image':<30} | {'Alt (m)':<8} | {'GSD (m/px)':<12} | {'IoU (%)':<10}")
    print("-" * 70)

    for img_name in images:
        img_path = os.path.join(img_folder, img_name)
        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        pose = extractor.extract_pose(img_bytes)
        if pose is None:
            continue

        # Decode for shape only or just use placeholder since we only want metrics
        # Actually mapper.feed needs the frame
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
        # REMOVED: cv2.rotate. Stay in sync with metadata orientation.

        # Connect mapper to pose_extractor which now stores metadata
        mapper.pose_extractor_ref = extractor

        # Feed to mapper which now prints internal metrics and registration info
        mapper.feed(frame, pose, K, metadata=extractor.last_metadata)

if __name__ == "__main__":
    analyze()
