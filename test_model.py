import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_NAME = "facebook/detr-resnet-50"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

# 1. Imagen
image = Image.open("test.jpg").convert("RGB")

# 2. Procesador y modelo (se guardan en ./models)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir="models")
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME, cache_dir="models")
model.to(device)
model.eval()

# 3. Preprocesamiento: imagen -> tensores, y a la GPU
inputs = processor(images=image, return_tensors="pt").to(device)

# 4. Inferencia
start = time.perf_counter()
with torch.no_grad():
    outputs = model(**inputs)
if device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"Tiempo de inferencia: {elapsed * 1000:.1f} ms")

# 5. Postprocesamiento: salida cruda -> objetos en píxeles
target_sizes = torch.tensor([image.size[::-1]])  # (alto, ancho)
results = processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    name = model.config.id2label[label.item()]
    coords = [round(v, 1) for v in box.tolist()]
    print(f"{name}: {score.item():.3f} en {coords}")
