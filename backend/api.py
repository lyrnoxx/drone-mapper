from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from .mapper_service import MapperService

app = FastAPI(title="DJI Mapper API")

# Enable CORS for React frontend (usually port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = MapperService.get_instance()
connected_clients = []

@app.get("/")
def read_root():
    return {"status": "Mapper API Running"}

@app.post("/feed-image")
async def feed_image(file: UploadFile = File(...)):
    """Receive an image file from the drone/simulation."""
    contents = await file.read()
    result = service.process_image(contents)
    
    # Notify WebSocket clients that map has updated
    if result["status"] == "success":
        await notify_clients("map_updated")
        
    return result

@app.get("/map/latest")
def get_map():
    """Get the current map as a JPEG image."""
    img_bytes = service.get_map_image()
    if img_bytes is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    return Response(content=img_bytes, media_type="image/jpeg")

@app.post("/reset")
def reset_map():
    service.reset_map()
    return {"status": "Map reset"}

async def notify_clients(message: str):
    for client in connected_clients:
        try:
            await client.send_text(message)
        except:
            pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Keep connection alive, maybe listen for client commands
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
