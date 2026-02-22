import os
import time
import requests
import sys

# CONFIG
API_URL = "http://127.0.0.1:8000/feed-image"
RGB_FOLDER = "images-true"
DELAY = 2.0

def run_simulation():
    if not os.path.exists(RGB_FOLDER):
        print(f"Error: Folder '{RGB_FOLDER}' not found.")
        return

    images = sorted([f for f in os.listdir(RGB_FOLDER) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not images:
        print("No images found.")
        return

    print(f"Starting HTTP Simulation with {len(images)} images...")
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(RGB_FOLDER, img_name)
        
        try:
            with open(img_path, 'rb') as f:
                files = {'file': (img_name, f, 'image/jpeg')}
                print(f"Sending {img_name} ({i+1}/{len(images)})...", end=" ")
                
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    print(f"Success: {response.json()}")
                else:
                    print(f"Failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            print(f"Error sending image: {e}")
            
        time.sleep(DELAY)

if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
