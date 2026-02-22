import numpy as np
from backend.core import PoseExtractor
import re

def test_metadata_extraction():
    print("Testing Metadata Extraction (Case Sensitivity)...")
    extractor = PoseExtractor()
    
    # Dummy XMP data with mixed case
    xmp_data = b"""
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/">
                <drone-dji:GpsLatitude>+28.123456</drone-dji:GpsLatitude>
                <drone-dji:GpsLongitude>+77.123456</drone-dji:GpsLongitude>
                <drone-dji:RelativeAltitude>+50.5</drone-dji:RelativeAltitude>
                <drone-dji:GimbalRollDegree>+1.2</drone-dji:GimbalRollDegree>
                <drone-dji:GimbalPitchDegree>-90.0</drone-dji:GimbalPitchDegree>
                <drone-dji:GimbalYawDegree>+45.0</drone-dji:GimbalYawDegree>
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>
    """
    
    # We need to simulate the JPEG structure enough for get_meta to find the segment
    # but get_meta just does .find(b'<x:xmpmeta')
    meta = extractor.get_meta(xmp_data)
    
    if meta:
        print(f"  Extracted: {meta}")
        assert abs(meta['GPSLatitude'] - 28.123456) < 1e-6
        assert abs(meta['GimbalPitchDegree'] - (-90.0)) < 1e-6
        print("  SUCCESS: Metadata extraction is case-insensitive.")
    else:
        print("  FAILED: Could not extract metadata.")

def test_rotation_logic():
    print("\nTesting Rotation Logic (Nadir/Horizontal/Yaw)...")
    extractor = PoseExtractor()
    
    # Helper to check forward vector in world frame
    def check_forward(yaw, pitch, roll, expected):
        m = {
            'GimbalYawDegree': yaw,
            'GimbalPitchDegree': pitch,
            'GimbalRollDegree': roll
        }
        rot = extractor.compute_camera_pose(m)
        # Forward in CV camera is +Z [0,0,1]
        f_world = rot.as_matrix() @ np.array([0, 0, 1])
        f_world = np.round(f_world, 3)
        expected = np.round(np.array(expected), 3)
        
        match = np.allclose(f_world, expected)
        status = "PASS" if match else "FAIL"
        print(f"  Yaw={yaw}, Pitch={pitch} -> Forward {f_world} | Expected {expected} | {status}")
        return match

    tests = [
        (0, -90, 0, [0, 0, 1]),    # Nadir (Down)
        (0, 0, 0, [1, 0, 0]),     # Horizontal North
        (90, 0, 0, [0, 1, 0]),    # Horizontal East
        (180, 0, 0, [-1, 0, 0]),  # Horizontal South
        (90, -90, 0, [0, 0, 1]),   # Nadir (Down, even with Yaw)
    ]
    
    results = [check_forward(*t) for t in tests]
    if all(results):
        print("  SUCCESS: Rotation logic matches NED expectations.")
    else:
        print("  FAILED: Rotation logic discrepancy.")

if __name__ == "__main__":
    try:
        test_metadata_extraction()
        test_rotation_logic()
    except Exception as e:
        print(f"Error during verification: {e}")
