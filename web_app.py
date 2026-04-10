import random
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Image Classifier")

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
