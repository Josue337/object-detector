import logging
import threading
import time

import torch
from PIL import Image

from model import load_model

logger = logging.getLogger(__name__)


class GPUOutOfMemoryError(RuntimeError):
    """La GPU no tiene memoria suficiente para procesar la imagen."""


class ObjectDetector:
    def __init__(self):
        self.processor, self.model, self.device, self.dtype = load_model()
        self._lock = threading.Lock()
        self._warmup()

    @property
    def precision(self) -> str:
        return "fp16" if self.dtype == torch.float16 else "fp32"

    def _warmup(self, runs: int = 2) -> None:
        dummy = Image.new("RGB", (640, 480))
        for _ in range(runs):
            self.detect(dummy)
        logger.info("Calentamiento completado")

    def detect(self, image: Image.Image, threshold: float = 0.9) -> dict:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        try:
            with self._lock:
                start = time.perf_counter()
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - start) * 1000
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise GPUOutOfMemoryError("La GPU se quedó sin memoria.") from exc

        # El postprocesamiento se hace en FP32 para no perder precisión en las cajas
        outputs.logits = outputs.logits.float()
        outputs.pred_boxes = outputs.pred_boxes.float()

        target_sizes = torch.tensor([image.size[::-1]])  # (alto, ancho)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        detections = [
            {
                "label": self.model.config.id2label[label.item()],
                "confidence": round(score.item(), 4),
                "box": [round(v, 1) for v in box.tolist()],
            }
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            )
        ]

        return {
            "device": self.device,
            "precision": self.precision,
            "inference_ms": round(inference_ms, 1),
            "detections": detections,
        }