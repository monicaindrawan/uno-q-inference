import os
import random
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from bluetooth_peer_node import PeerNode, load_peer_macs

peer_node = PeerNode(
    my_name=socket.gethostname(),
    peer_macs=load_peer_macs("peers.json"),
    channel=4,
    interval=5,
    fixed_message="PING",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    peer_node.start()
    yield
    peer_node.stop()


app = FastAPI(title="Image Classifier", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def mock_classify(image_bytes: bytes) -> int:
    """Mock AI model — replace with real inference later."""
    return random.randint(0, 9)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext != ".ppm":
        return {"error": f"Unsupported file type '{ext}'. Only .ppm files are accepted."}

    image_bytes = await file.read()
    predicted_class = mock_classify(image_bytes)

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
    }


@app.get("/peer-status")
async def peer_status():
    return {
        "connected": peer_node.is_connected,
        "active_peer_mac": peer_node.active_peer_mac,
        "candidate_peer_macs": peer_node.peer_macs,
        "messages_sent": peer_node.messages_sent,
        "messages_received": peer_node.messages_received,
        "messages_exchanged": peer_node.messages_exchanged,
    }
