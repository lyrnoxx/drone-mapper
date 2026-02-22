import numpy as np
from scipy.spatial.transform import Rotation as R

print("\n--- Testing Z-Y-X Intrinsic with Corrected Pitch Sign ---")
def test_correct(y, p_dji, r, seq='zyx'):
    p = -p_dji
    rot = R.from_euler(seq.upper(), [y, p, r], degrees=True)
    matrix = rot.as_matrix()
    forward = matrix @ np.array([0,0,1]) # This is where North WAS.
    # Wait, in CV frame, Forward is Z. If Pitch=0, Yaw=0, Z should be North [1,0,0].
    # But R.from_euler('ZYX', [0,0,0]) is Identity, so Z is [0,0,1].
    # This means the "default" camera in R is pointing at World Z (Down).
    # That explains why my ZYX test gave Forward = Down for Pitch 0!
    
    # Let's try base orientation: Camera Z points at World X (North).
    # Then rotate.
    # OR: Better to define the camera in World frame:
    # R_cam_to_world = R_yaw * R_pitch * R_roll
    
    r_yaw = R.from_euler('z', y, degrees=True)
    r_pitch = R.from_euler('y', -p_dji, degrees=True) # North to Down is +90 deg around Y
    r_roll = R.from_euler('x', r, degrees=True)
    
    # Total rotation: First Yaw, then Pitch, then Roll
    # Actually, DJI gimbal is World -> Yaw -> Pitch -> Roll.
    # So R = R_yaw * R_pitch * R_roll (Intrinsic)
    rot_total = r_yaw * r_pitch * r_roll
    matrix = rot_total.as_matrix()
    
    # Now, in the Camera frame (after rotation), where is the World X, Y, Z?
    # Or, where is the Camera Forward (Local Z) in World frame?
    # Important: In CV standard, Forward is Z.
    # Wait, if the camera starts pointing at North (X), then its local Z is [1,0,0].
    # No, local Z is ALWAYS [0,0,1]. If it points at North, then R takes [0,0,1] to [1,0,0].
    
    # If Pitch = 0, Yaw = 0, Forward should be [1,0,0].
    # If Pitch = -90, Yaw = 0, Forward should be [0,0,1].
    
    # Let's check R_pitch = R.from_euler('y', 90)
    # R_pitch * [0,0,1] = [1,0,0]. Wait. Right hand rule Y: fingers from Z to X is positive.
    # So 90 deg around Y takes Z to X.
    # So if Pitch=0 (pointing North), we want Z to be X.
    # If Pitch=-90 (pointing Down), we want Z to be Z.
    
    # CONCLUSION: The base orientation of the CV camera (Z forward) matches Nadir if identity!
    # If R = Identity, Camera Z = World Z = Down.
    # If so, DJI Pitch -90 (Nadir) should have NO pitch rotation.
    # DJI Pitch 0 (Horizontal) should have -90 pitch rotation?
    
    # Let's try:
    # R_cam = R_yaw * R_pitch_corrected
    # pitch_corrected = metadata['GimbalPitchDegree'] + 90
    # If DJI_Pitch = -90, pitch_corrected = 0. (Identity) -> Z is Down.
    # If DJI_Pitch = 0,   pitch_corrected = 90. (90 around Y) -> Z is North.
    
    res_rot = R.from_euler('z', y, degrees=True) * R.from_euler('y', p_dji + 90, degrees=True)
    f = res_rot.as_matrix() @ np.array([0,0,1])
    print(f"Yaw={y}, Pitch={p_dji} -> Forward={f.round(3)}")

test_correct(0, -90, 0) # Expected [0,0,1]
test_correct(0, 0, 0)   # Expected [1,0,0]
test_correct(90, -90, 0) # Expected [0,0,1]
test_correct(90, 0, 0)   # Expected [0,1,0]
