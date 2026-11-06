from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_NAME = "facebook/detr-resnet-50"
CACHE_DIR = Path(__file__).resolve().parent.parent / "models"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = MODEL_NAME):
    device = get_device()
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForObjectDetection.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model.to(device)
    model.eval()
    return processor, model, device
