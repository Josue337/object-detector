from PIL import Image

from detector import ObjectDetector

detector = ObjectDetector()
image = Image.open("../test.jpg")

for i in range(3):
    result = detector.detect(image)
    print(f"Ejecución {i + 1}: {result['inference_ms']} ms, {len(result['detections'])} objetos")

print(result)
