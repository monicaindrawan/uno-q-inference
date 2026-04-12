from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from inference import collaborative_inference, fusion_inference, get_peer_status, solo_inference


app = FastAPI(title="Image Classifier")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/classify")
async def classify(file: UploadFile = File(...), method: str = "collaborative"):
    ext = Path(file.filename).suffix.lower()
    if ext != ".ppm":
        return {"error": f"Unsupported file type '{ext}'. Only .ppm files are accepted."}

    image_bytes = await file.read()

    if method == "fusion":
        output = fusion_inference(image_bytes)
    elif method == "solo":
        output = solo_inference(image_bytes)
    else:
        output = collaborative_inference(image_bytes)

    return {
        "filename": file.filename,
        "method": output["method"],
        "pred_class": output["pred_class"],
        "reason": output.get("reason"),
        "confidence": output.get("confidence")
    }


@app.get("/peer-status")
async def peer_status():
    return get_peer_status()
