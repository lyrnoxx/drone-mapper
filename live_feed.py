"""
Live Feed Script for DJI M350 RTK + MK15

Watches a folder for new DJI images (from drone/SD card/MK15)
and feeds them to the mapper server in real-time.

Usage:
  1. Start the mapper server:    streamlit run dji_mapper_app.py
  2. Start this script:          python live_feed.py --watch <folder>

Modes:
  --watch <folder>   Watch a local folder for new .jpg files and auto-POST
  --adb              Watch MK15 controller storage via ADB (USB connected)
  --replay <folder>  Replay existing images with realistic timing
"""

import os
import sys
import time
import argparse
import requests
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

API_URL = "http://127.0.0.1:8000/feed-image"


def post_image(filepath, api_url=API_URL):
    """POST an image to the mapper server."""
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f, 'image/jpeg')}
            resp = requests.post(api_url, files=files, timeout=30)
            return resp.status_code == 200, resp.json() if resp.ok else resp.text
    except requests.ConnectionError:
        return False, "Server not running"
    except Exception as e:
        return False, str(e)


def watch_folder(folder, api_url=API_URL, interval=0.5):
    """
    Watch a folder for new JPEG files and auto-POST them.
    
    This is the simplest deployment mode:
    1. Set drone to interval shooting (2s)
    2. Copy images to this folder (manually, FTP, or auto-sync)
    3. Script picks them up and feeds the mapper
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"Error: Folder '{folder}' does not exist")
        sys.exit(1)

    seen = set()
    # Pre-populate with existing files so we don't re-process old images
    for f in folder.glob("*.jpg"):
        seen.add(f.name)
    for f in folder.glob("*.jpeg"):
        seen.add(f.name)
    for f in folder.glob("*.JPG"):
        seen.add(f.name)

    print(f"[Live Feed] Watching: {folder}")
    print(f"[Live Feed] Server:   {api_url}")
    print(f"[Live Feed] Existing files skipped: {len(seen)}")
    print(f"[Live Feed] Waiting for new images... (Ctrl+C to stop)\n")

    total_sent = 0
    try:
        while True:
            # Scan for new files
            current_files = set()
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
                for f in folder.glob(ext):
                    current_files.add(f.name)

            new_files = sorted(current_files - seen)

            for fname in new_files:
                filepath = folder / fname
                
                # Wait for file to be fully written (size stable for 0.5s)
                size1 = filepath.stat().st_size
                time.sleep(0.3)
                size2 = filepath.stat().st_size
                if size1 != size2:
                    continue  # Still being written
                
                # Verify it's a valid JPEG (at least 10KB)
                if size2 < 10240:
                    print(f"  [Skip] {fname} too small ({size2} bytes)")
                    seen.add(fname)
                    continue

                total_sent += 1
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"  [{ts}] #{total_sent} Feeding {fname} ({size2/1024:.0f} KB)...", end=" ")
                
                ok, result = post_image(filepath, api_url)
                if ok:
                    print(f"OK")
                else:
                    print(f"FAIL: {result}")
                
                seen.add(fname)

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n[Live Feed] Stopped. Total images sent: {total_sent}")


def watch_adb(api_url=API_URL, poll_interval=2.0):
    """
    Watch the MK15 controller's storage for new photos via ADB.
    
    The MK15 stores downloaded media at:
      /storage/emulated/0/DJI/com.dji.industry.pilot/mediafiles/
    
    Requirements:
      - ADB installed and in PATH
      - MK15 connected via USB with USB debugging enabled
      - DJI Pilot 2 configured to auto-download photos
    """
    # DJI Pilot 2 media paths on the controller
    MEDIA_PATHS = [
        "/storage/emulated/0/DJI/com.dji.industry.pilot/mediafiles/",
        "/sdcard/DJI/com.dji.industry.pilot/mediafiles/",
        "/storage/emulated/0/DCIM/DJI/",
        "/sdcard/DCIM/DJI/",
    ]
    
    # Check ADB is available
    try:
        result = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=5
        )
        devices = [l for l in result.stdout.strip().split('\n')[1:] if l.strip()]
        if not devices:
            print("Error: No ADB devices found. Is MK15 connected via USB?")
            print("  1. Enable USB Debugging on MK15 (Settings > Developer Options)")
            print("  2. Connect MK15 to PC via USB-C")
            print("  3. Accept the debugging prompt on MK15")
            sys.exit(1)
        print(f"[ADB] Connected device: {devices[0].split()[0]}")
    except FileNotFoundError:
        print("Error: ADB not found. Install Android Platform Tools:")
        print("  https://developer.android.com/tools/releases/platform-tools")
        print("  Then add to PATH")
        sys.exit(1)
    
    # Find which media path exists on the controller
    active_path = None
    for p in MEDIA_PATHS:
        result = subprocess.run(
            ["adb", "shell", f"ls {p}"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "No such file" not in result.stderr:
            active_path = p
            print(f"[ADB] Media path: {active_path}")
            break
    
    if not active_path:
        print("[ADB] Could not find DJI media folder on controller.")
        print("[ADB] Available folders:")
        subprocess.run(["adb", "shell", "ls /sdcard/DJI/"], timeout=5)
        print("\nTry flying and taking a photo first, then re-run this script.")
        sys.exit(1)
    
    # Create temp folder for pulled images
    tmp_dir = Path(tempfile.mkdtemp(prefix="dji_live_"))
    print(f"[ADB] Temp folder: {tmp_dir}")
    print(f"[ADB] Polling every {poll_interval}s for new photos...\n")
    
    seen = set()
    total_sent = 0
    
    try:
        while True:
            # List files on the controller
            result = subprocess.run(
                ["adb", "shell", f"find {active_path} -name '*.JPG' -o -name '*.jpg' 2>/dev/null"],
                capture_output=True, text=True, timeout=10
            )
            
            remote_files = [
                l.strip() for l in result.stdout.strip().split('\n')
                if l.strip() and l.strip().endswith(('.JPG', '.jpg'))
            ]
            
            new_files = sorted(set(remote_files) - seen)
            
            for remote_path in new_files:
                fname = os.path.basename(remote_path)
                local_path = tmp_dir / fname
                
                # Pull file from controller
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"  [{ts}] Pulling {fname}...", end=" ")
                
                pull_result = subprocess.run(
                    ["adb", "pull", remote_path, str(local_path)],
                    capture_output=True, text=True, timeout=30
                )
                
                if pull_result.returncode != 0:
                    print(f"FAIL: {pull_result.stderr}")
                    seen.add(remote_path)
                    continue
                
                # POST to mapper
                total_sent += 1
                size = local_path.stat().st_size
                print(f"({size/1024:.0f} KB) → POST #{total_sent}...", end=" ")
                
                ok, result_msg = post_image(str(local_path), api_url)
                if ok:
                    print("OK")
                else:
                    print(f"FAIL: {result_msg}")
                
                # Clean up local copy to save disk
                local_path.unlink(missing_ok=True)
                seen.add(remote_path)
            
            time.sleep(poll_interval)
    
    except KeyboardInterrupt:
        print(f"\n[ADB] Stopped. Total images sent: {total_sent}")
        # Clean up temp dir
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except:
            pass


def replay_folder(folder, api_url=API_URL, delay=2.0):
    """
    Replay existing images with a delay (simulates real flight timing).
    Like sim_http.py but with better feedback.
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"Error: Folder '{folder}' does not exist")
        sys.exit(1)

    images = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg')
    ])

    if not images:
        print(f"No JPEG images found in {folder}")
        sys.exit(1)

    print(f"[Replay] {len(images)} images from {folder}")
    print(f"[Replay] Delay: {delay}s between images")
    print(f"[Replay] Server: {api_url}")
    print(f"[Replay] Estimated time: {len(images) * delay:.0f}s\n")

    for i, img_path in enumerate(images):
        ts = datetime.now().strftime('%H:%M:%S')
        size = img_path.stat().st_size
        print(f"  [{ts}] {i+1}/{len(images)} {img_path.name} ({size/1024:.0f} KB)...", end=" ")
        
        ok, result = post_image(str(img_path), api_url)
        if ok:
            print("OK")
        else:
            print(f"FAIL: {result}")
        
        if i < len(images) - 1:
            time.sleep(delay)

    print(f"\n[Replay] Done. {len(images)} images fed to mapper.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live feed script for DJI M350 RTK → Mapper Server"
    )
    parser.add_argument(
        "--watch", metavar="FOLDER",
        help="Watch a folder for new images and auto-POST them"
    )
    parser.add_argument(
        "--adb", action="store_true",
        help="Watch MK15 controller storage via ADB"
    )
    parser.add_argument(
        "--replay", metavar="FOLDER",
        help="Replay existing images with realistic timing"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay between images in replay mode (default: 2.0s)"
    )
    parser.add_argument(
        "--server", default=API_URL,
        help=f"Mapper server URL (default: {API_URL})"
    )

    args = parser.parse_args()

    if args.watch:
        watch_folder(args.watch, api_url=args.server)
    elif args.adb:
        watch_adb(api_url=args.server)
    elif args.replay:
        replay_folder(args.replay, api_url=args.server, delay=args.delay)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python live_feed.py --watch C:\\DronePhotos")
        print("  python live_feed.py --adb")
        print("  python live_feed.py --replay images-true --delay 1.5")
