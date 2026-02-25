"""
MAVLink-to-Mapper Live Feed for Pixhawk + Skydroid MK15

Bridges MAVLink telemetry from Pixhawk (via MK15) and camera frames
into DJI-compatible geotagged JPEGs that the mapper server accepts.

Architecture:
  Pixhawk → Skydroid Air Unit → MK15 (ttyHS1) → ADB forward → PC (this script)
  Skydroid Camera → MK15 HDMI out → HDMI capture card → OpenCV (this script)

Usage:
  1. Connect MK15 via USB-C, drone powered on
  2. python mavlink_feed.py --setup        (one-time ADB port forward)
  3. python mavlink_feed.py --live          (start live feed)
  4. python mavlink_feed.py --test-telem    (test telemetry only, no camera)

Requirements:
  pip install pymavlink opencv-python requests
"""

import os
import sys
import io
import time
import struct
import argparse
import threading
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

# ── Configuration ──
API_URL = "http://127.0.0.1:8000/feed-image"
ADB_PATH = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "platform-tools", "adb.exe")

# MAVLink connection options (tried in order)
MAVLINK_CONNECTIONS = [
    "tcp:127.0.0.1:14445",       # ADB-forwarded TCP from MK15
    "tcp:127.0.0.1:5760",        # Alternative MAVLink port
    "udp:127.0.0.1:14550",       # UDP variant
    "udp:0.0.0.0:14551",         # QGC forwarded output
]

# Camera capture options
CAMERA_INDEX = 0  # OpenCV camera index (HDMI capture card)

# Capture interval
CAPTURE_INTERVAL_S = 2.0  # seconds between captures

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# MAVLink Telemetry Reader
class MAVLinkTelemetry:
    """
    Reads GPS position and attitude from Pixhawk via MAVLink.
    Provides the same data that DJI XMP metadata contains:
    - Latitude, Longitude, Altitude
    - Gimbal/Flight Roll, Pitch, Yaw
    """
    def __init__(self):
        self.latest = {
            'lat': 0.0,
            'lon': 0.0,
            'alt_rel': 0.0,
            'alt_abs': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'gps_fix': 0,
            'satellites': 0,
            'hdop': 999,
            'timestamp': 0,
        }
        self.conn = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._msg_count = 0

    def connect(self, connection_string=None):
        """
        Try to establish a MAVLink connection using the known endpoints.
        Does not start the reader thread by itself.
        """
        try:
            from pymavlink import mavutil
        except ImportError:
            log.error("pymavlink not installed. Run: pip install pymavlink")
            return False

        endpoints = [connection_string] if connection_string else MAVLINK_CONNECTIONS
        for ep in endpoints:
            try:
                log.info(f"Trying MAVLink connection: {ep}")
                self.conn = mavutil.mavlink_connection(ep, baud=57600, source_system=255)
                log.info("Waiting for heartbeat...")
                msg = self.conn.wait_heartbeat(timeout=5)
                if msg:
                    log.info(
                        f"Heartbeat from system {self.conn.target_system}, "
                        f"component {self.conn.target_component}"
                    )
                    log.info(f"Autopilot: {msg.autopilot}, Type: {msg.type}")
                    return True
                else:
                    log.warning(f"No heartbeat on {ep}")
                    self.conn = None
            except Exception as e:
                log.warning(f"Failed to connect to {ep}: {e}")
                self.conn = None
        return False

    def start(self):
        """
        Start background MAVLink message reader.
        """
        if not self.conn:
            ok = self.connect()
            if not ok:
                return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("MAVLink telemetry reader started.")
        return True

    def stop(self):
        """
        Stop background reader and close connection.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        log.info("MAVLink telemetry reader stopped.")

    def _loop(self):
        """
        Internal loop: read MAVLink messages and update latest telemetry.
        """
        try:
            from pymavlink import mavutil
        except ImportError:
            log.error("pymavlink not installed. Run: pip install pymavlink")
            self._running = False
            return

        while self._running and self.conn:
            try:
                msg = self.conn.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue

                self._msg_count += 1
                mtype = msg.get_type()

                with self._lock:
                    # GPS position
                    if mtype == "GLOBAL_POSITION_INT":
                        self.latest["lat"] = msg.lat / 1e7
                        self.latest["lon"] = msg.lon / 1e7
                        # alt in millimeters
                        self.latest["alt_abs"] = msg.alt / 1000.0
                        self.latest["alt_rel"] = msg.relative_alt / 1000.0
                        self.latest["vx"] = msg.vx / 100.0
                        self.latest["vy"] = msg.vy / 100.0
                        self.latest["vz"] = msg.vz / 100.0
                        self.latest["timestamp"] = time.time()

                    # Attitude
                    elif mtype == "ATTITUDE":
                        self.latest["roll"] = msg.roll * 57.2957795
                        self.latest["pitch"] = msg.pitch * 57.2957795
                        self.latest["yaw"] = msg.yaw * 57.2957795
                        self.latest["timestamp"] = time.time()

                    # GPS status
                    elif mtype == "GPS_RAW_INT":
                        self.latest["gps_fix"] = msg.fix_type
                        self.latest["satellites"] = msg.satellites_visible
                        # hdop in cm; convert to m
                        if hasattr(msg, "eph") and msg.eph is not None:
                            self.latest["hdop"] = msg.eph / 100.0

            except Exception as e:
                log.debug(f"MAVLink read error: {e}")
                continue

# ═══════════════════════════════════════════════════════════
# Camera Capture
class CameraCapture:
    """
    Captures frames from HDMI capture card using OpenCV.
    """
    def __init__(self, index=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            log.warning("Failed to capture frame from camera.")
            return None
        return frame
    def release(self):
        self.cap.release()


def build_dji_xmp(telemetry):
    """
    Build DJI-compatible XMP XML from telemetry dict.
    """
    xmp = f"""
    <x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:drone-dji='http://www.dji.com/drone-dji/1.0/'>
      <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
        <rdf:Description rdf:about=''
          drone-dji:AbsoluteAltitude='{telemetry['alt_abs']}'
          drone-dji:RelativeAltitude='{telemetry['alt_rel']}'
          drone-dji:Latitude='{telemetry['lat']}'
          drone-dji:Longitude='{telemetry['lon']}'
          drone-dji:GimbalPitchDegree='-90.0'
          drone-dji:GimbalYawDegree='{telemetry['yaw']}'
          drone-dji:GimbalRollDegree='{telemetry['roll']}'
          drone-dji:FlightPitchDegree='{telemetry['pitch']}'
          drone-dji:FlightRollDegree='{telemetry['roll']}'
          drone-dji:FlightYawDegree='{telemetry['yaw']}'
        />
      </rdf:RDF>
    </x:xmpmeta>
    """
    return xmp.strip()


def inject_xmp_into_jpeg(jpeg_bytes, xmp_str):
    """
    Inject XMP metadata as APP1 marker after JPEG SOI.
    """
    soi = b'\xff\xd8'
    app1_marker = b'\xff\xe1'
    xmp_bytes = xmp_str.encode('utf-8')
    app1_len = len(xmp_bytes) + 2 + 29
    app1_header = app1_marker + struct.pack('>H', app1_len) + b'http://ns.adobe.com/xap/1.0/\x00' + xmp_bytes
    if jpeg_bytes[:2] != soi:
        raise ValueError("Not a JPEG file")
    return soi + app1_header + jpeg_bytes[2:]


def frame_to_geotagged_jpeg(frame, telemetry):
    """
    Convert OpenCV frame to JPEG and inject XMP.
    """
    _, jpeg_bytes = cv2.imencode('.jpg', frame)
    xmp = build_dji_xmp(telemetry)
    return inject_xmp_into_jpeg(jpeg_bytes.tobytes(), xmp)


def run_live_feed():
    telem = MAVLinkTelemetry()
    if not telem.connect():
        log.error("Could not connect to MAVLink telemetry.")
        return
    cam = CameraCapture()
    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            with telem._lock:
                telemetry = telem.latest.copy()
            jpeg = frame_to_geotagged_jpeg(frame, telemetry)
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            fname = f"live_{ts}.jpg"
            log.info(f"Posting image {fname} with telemetry {telemetry}")
            try:
                resp = requests.post(API_URL, files={'file': (fname, jpeg, 'image/jpeg')})
                log.info(f"POST result: {resp.status_code}")
            except Exception as e:
                log.error(f"POST failed: {e}")
            time.sleep(CAPTURE_INTERVAL_S)
    finally:
        cam.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAVLink-to-Mapper Live Feed")
    parser.add_argument("--setup", action="store_true", help="Setup ADB port forwarding")
    parser.add_argument("--live", action="store_true", help="Start live feed (telemetry + camera)")
    parser.add_argument("--test-telem", action="store_true", help="Test MAVLink telemetry only")
    parser.add_argument("--test-camera", action="store_true", help="Test camera capture only")
    args = parser.parse_args()
    if args.setup:
        cmd = f'"{ADB_PATH}" forward tcp:14445 tcp:14445'
        log.info(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)
        log.info("ADB port forwarding set up.")
    elif args.live:
        run_live_feed()
    elif args.test_telem:
        telem = MAVLinkTelemetry()
        if telem.connect():
            log.info("Telemetry connection successful.")
            log.info(f"Latest telemetry: {telem.latest}")
        else:
            log.error("Telemetry connection failed.")
    elif args.test_camera:
        cam = CameraCapture()
        frame = cam.get_frame()
        if frame is not None:
            cv2.imshow("Camera Test", frame)
            cv2.waitKey(2000)
        cam.release()
        log.info("Camera test complete.")
