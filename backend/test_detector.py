import torch
from PIL import Image

from detector import ObjectDetector

detector = ObjectDetector()
image = Image.open("../test.jpg")

times = [detector.detect(image)["inference_ms"] for _ in range(30)]
result = detector.detect(image)

print(f"Precisión: {detector.precision}")
print(f"Media: {sum(times) / len(times):.1f} ms | mínimo: {min(times):.1f} ms")
print([(d["label"], d["confidence"]) for d in result["detections"]])
if detector.device == "cuda":
    print(f"VRAM pico: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")