import logging
import os
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/detr-resnet-50")
CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
USE_FP16 = os.getenv("USE_FP16", "1") == "1"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = MODEL_NAME):
    device = get_device()
    dtype = torch.float16 if device == "cuda" and USE_FP16 else torch.float32

    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForObjectDetection.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model.to(device=device, dtype=dtype)
    model.eval()

    logger.info("Modelo %s cargado en %s (%s)", model_name, device, dtype)
    return processor, model, device, dtype