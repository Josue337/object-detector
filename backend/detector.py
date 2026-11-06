import time

import torch
from PIL import Image

from model import load_model


class ObjectDetector:
    def __init__(self):
        self.processor, self.model, self.device = load_model()

    def detect(self, image: Image.Image, threshold: float = 0.9) -> dict:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000

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
            "inference_ms": round(inference_ms, 1),
            "detections": detections,
        }
