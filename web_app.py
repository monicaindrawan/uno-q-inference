import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from inference import fusion_inference, get_peer_status, solo_inference


app = FastAPI(title="Image Classifier")
templates = Jinja2Templates(directory="templates")


def solo_classify(image_bytes: bytes) -> int: 
    return {
        "method": "Solo Inference",
        "predicted_class": solo_inference(image_bytes)
    }


def fusion_classify(image_bytes: bytes) -> int:
    return {
        "method": "Fusion Inference",
        "predicted_class": fusion_inference(image_bytes)
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext != ".ppm":
        return {"error": f"Unsupported file type '{ext}'. Only .ppm files are accepted."}

    image_bytes = await file.read()
    output = solo_classify(image_bytes)

    return {
        "filename": file.filename,
        "method": output["method"],
        "predicted_class": output["predicted_class"],
    }


@app.get("/peer-status")
async def peer_status():
    return get_peer_status()
