"""
Web App — FastAPI Node Server
==============================

Exposes three HTTP endpoints for this camera node:

    GET  /              — serves the Jinja2 UI (index.html) with the node name
                          and default merge operator injected as template vars
    POST /classify      — receives a .ppm image and runs inference in one of
                          three modes (solo / fusion / collaborative), returning
                          the predicted class, confidence, method, latency, and
                          a reasoning explaining the decision (for fusion/collaborative)
    GET  /peer-status   — returns the current Bluetooth peer connection state
                          (used by the UI to poll and update the status banner)

Inference logic lives entirely in inference.py; this file is only routing +
request/response handling.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from inference import NODE_NAME, collaborative_inference, fusion_inference, get_peer_status, solo_inference


# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(title="Image Classifier")
templates = Jinja2Templates(directory="templates")


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the classifier UI, injecting this node's name into the template."""
    return templates.TemplateResponse(request, "index.html", {"node_name": NODE_NAME, "merge_operator": "fusion_head"})


@app.post("/classify")
async def classify(file: UploadFile = File(...), method: str = "collaborative", merge_operator: str = 'fusion_head'):
    """Run inference on an uploaded .ppm image.

    Query params:
        method         : "solo" | "fusion" | "collaborative" (default: collaborative)
        merge_operator : strategy used when combining peer outputs
                         (e.g. "fusion_head", "confidence_weighted_mean")

    Returns a JSON object with:
        filename     : original uploaded filename
        method       : inference method actually used (may differ if peer unavailable)
        pred_class   : predicted GTSRB class ID
        confidence   : softmax confidence of the top prediction (float 0–1)
        reason       : human-readable explanation (fusion/collaborative only)
        extra_info   : dict with solo_pred_class etc. (collaborative only)
        inference_ms : end-to-end server-side latency in milliseconds
    """
    ext = Path(file.filename).suffix.lower()
    if ext != ".ppm":
        return {"error": f"Unsupported file type '{ext}'. Only .ppm files are accepted."}

    image_bytes = await file.read()

    # Dispatch to the appropriate inference function and time it
    t0 = time.perf_counter()
    if method == "fusion":
        output = fusion_inference(image_bytes, merge_operator=merge_operator)
    elif method == "solo":
        output = solo_inference(image_bytes)
    else:
        output = collaborative_inference(image_bytes, merge_operator=merge_operator)
    inference_ms = (time.perf_counter() - t0) * 1000

    return {
        "filename": file.filename,
        "method": output["method"],
        "pred_class": output["pred_class"],
        "reason": output.get("reason"),
        "confidence": output.get("confidence"),
        "extra_info": output.get("extra_info"),
        "inference_ms": round(inference_ms, 1),
    }


@app.get("/peer-status")
async def peer_status():
    """Return Bluetooth peer connection state for the UI status banner."""
    return get_peer_status()
