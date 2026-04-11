import os
import random
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from peer_node import PeerNode, load_peer_macs

# ---------------------------------------------------------------------------
# Bluetooth peer configuration — set via environment variables before launch:
#   PEERS_FILE   path to JSON file listing peers (default: peers.json)
#   BT_CHANNEL   RFCOMM channel                 (default: 4)
#   BT_INTERVAL  seconds between messages       (default: 5)
#   BT_MESSAGE   fixed message to send          (default: PING)
#
# peers.json format:
#   { "peers": [ {"name": "NODE_B", "mac": "AA:BB:CC:DD:EE:FF"}, ... ] }
# ---------------------------------------------------------------------------
_PEER_NAME = socket.gethostname()
_PEERS_FILE = os.getenv("PEERS_FILE", "peers.json")
_BT_CHANNEL = int(os.getenv("BT_CHANNEL", "4"))
_BT_INTERVAL = int(os.getenv("BT_INTERVAL", "5"))
_BT_MESSAGE = os.getenv("BT_MESSAGE", "PING")

peer_node = PeerNode(
    my_name=_PEER_NAME,
    peer_macs=load_peer_macs(_PEERS_FILE),
    channel=_BT_CHANNEL,
    interval=_BT_INTERVAL,
    fixed_message=_BT_MESSAGE,
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
