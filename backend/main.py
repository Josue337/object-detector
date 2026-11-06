import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from detector import ObjectDetector

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["detector"] = ObjectDetector()  # se ejecuta al arrancar
    yield
    ml.clear()  # se ejecuta al apagar


app = FastAPI(title="Object Detector API", version="1.0.0", lifespan=lifespan)


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]  # [x_min, y_min, x_max, y_max] en píxeles


class DetectionResponse(BaseModel):
    device: str
    inference_ms: float
    image_width: int
    image_height: int
    detections: list[Detection]


@app.get("/health")
def health():
    detector = ml.get("detector")
    return {
        "status": "ok" if detector else "loading",
        "device": detector.device if detector else None,
    }


@app.post("/detect", response_model=DetectionResponse)
def detect(
    file: UploadFile = File(...),
    threshold: float = Query(0.9, ge=0.0, le=1.0),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Formato no soportado. Usa JPEG, PNG o WebP.")

    data = file.file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="La imagen supera los 10 MB.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

    result = ml["detector"].detect(image, threshold=threshold)
    return DetectionResponse(
        **result, image_width=image.width, image_height=image.height
    )
