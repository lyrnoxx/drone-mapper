import socket
import struct
import time
import os
import threading
import sys

# CONFIG
HOST = '127.0.0.1'
PORT = 5006
RGB_FOLDER = "images-true"

class DroneReplay(threading.Thread):
    def __init__(self, drone_id, image_folder, delay=0.1):
        super().__init__()
        self.drone_id = drone_id
        self.image_folder = image_folder
        self.delay = delay
        self.sock = None

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((HOST, PORT))
            print(f"[{self.drone_id}] Connected to Ground Station.")
            return True
        except ConnectionRefusedError:
            print(f"[{self.drone_id}] Connection Failed! Is server running?")
            return False

    def run(self):
        if not os.path.exists(self.image_folder):
            print(f"[{self.drone_id}] Error: Image folder '{self.image_folder}' missing!")
            return
        
        # Retry connection
        while not self.connect():
            time.sleep(2)

        # Get list of images
        images = sorted([f for f in os.listdir(self.image_folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not images:
            print(f"[{self.drone_id}] No images found in '{self.image_folder}'!")
            return

        print(f"[{self.drone_id}] Starting Mission ({len(images)} images).")

        for i, img_name in enumerate(images):
            img_path = os.path.join(self.image_folder, img_name)

            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
            except Exception:
                continue

            img_len = len(img_data)

            # Protocol:
            # 1 byte  : 'I' (image only)
            # 4 bytes : image length
            header = struct.pack('<cI', b'I', img_len)

            try:
                self.sock.sendall(header + img_data)

                if i % 5 == 0:
                    print(f"[{self.drone_id}] Sent image {img_name} ({i+1}/{len(images)})")

            except BrokenPipeError:
                print(f"[{self.drone_id}] Server disconnected.")
                break
            except Exception as e:
                print(f"[{self.drone_id}] Socket Error: {e}")
                break

            time.sleep(self.delay)

        # End of mission
        try:
            self.sock.sendall(b'S')
            self.sock.close()
            print(f"[{self.drone_id}] Mission Complete.")
        except:
            pass

if __name__ == "__main__":
    if not os.path.exists(RGB_FOLDER):
        print(f"ERROR: '{RGB_FOLDER}' folder not found.")
        sys.exit(1)

    drone1 = DroneReplay("Alpha", RGB_FOLDER, delay=0.5)
    drone1.start()
    drone1.join()
