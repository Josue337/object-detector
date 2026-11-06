import io
import os

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 30

# Colores en RGB: la imagen se dibuja como array RGB, sin pasar a BGR
PALETTE = [
    (79, 139, 249), (239, 85, 59), (0, 204, 150), (171, 99, 250),
    (255, 161, 90), (25, 211, 243), (255, 102, 146), (182, 232, 128),
]

st.set_page_config(page_title="Detector de objetos", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.12);
        border-radius: 12px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_detect(image_bytes: bytes, filename: str, content_type: str, threshold: float) -> dict:
    response = requests.post(
        f"{API_URL}/detect",
        params={"threshold": threshold},
        files={"file": (filename, image_bytes, content_type)},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = np.array(image.convert("RGB"))
    height, width = canvas.shape[:2]
    scale = max(width, height) / 1000
    box_thickness = max(2, round(scale * 3))
    font_scale = max(0.5, scale * 0.8)
    text_thickness = max(1, box_thickness // 2)
    colors: dict[str, tuple] = {}

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["box"])
        x1, y1 = max(0, x1), max(0, y1)
        color = colors.setdefault(det["label"], PALETTE[len(colors) % len(PALETTE)])

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

        text = f"{det['label']} {det['confidence']:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        top = max(y1 - text_h - baseline - 6, 0)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 8, top + text_h + baseline + 6), color, -1)
        cv2.putText(
            canvas, text, (x1 + 4, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
        )

    return Image.fromarray(canvas)


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    threshold = st.slider("Umbral de confianza", 0.10, 0.99, 0.90, 0.01)
    st.caption("Solo se muestran objetos con confianza igual o mayor a este valor.")

    health = get_health()
    if health and health["status"] == "ok":
        device_name = "GPU (CUDA)" if health["device"] == "cuda" else "CPU"
        st.success(f"API conectada · {device_name}")
    else:
        st.error("API no disponible")
    st.caption(f"URL de la API: {API_URL}")

# ---------- Cabecera ----------
st.title("🔍 Detector de objetos")
st.caption("DETR ResNet-50 · FastAPI · PyTorch · CUDA")

# ---------- Entrada ----------
tab_upload, tab_camera = st.tabs(["📁 Subir imagen", "📷 Cámara"])
with tab_upload:
    uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    photo = st.camera_input("Toma una foto")

source = uploaded or photo
if source is None:
    st.info("Sube una imagen o toma una foto para empezar.")
    st.stop()

image_bytes = source.getvalue()
image = Image.open(io.BytesIO(image_bytes))

# ---------- Llamada a la API ----------
with st.spinner("Detectando objetos..."):
    try:
        result = call_detect(image_bytes, source.name, source.type, threshold)
    except requests.HTTPError as error:
        try:
            detail = error.response.json()["detail"]
        except Exception:
            detail = error.response.text
        st.error(f"La API rechazó la imagen: {detail}")
        st.stop()
    except requests.RequestException:
        st.error(f"No se pudo conectar con la API en {API_URL}. ¿Está el backend en ejecución?")
        st.stop()

detections = result["detections"]
annotated = draw_boxes(image, detections)

# ---------- Imágenes ----------
col_original, col_result = st.columns(2)
with col_original:
    st.subheader("Imagen original")
    st.image(image)
with col_result:
    st.subheader("Objetos detectados")
    st.image(annotated)

# ---------- Métricas ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Objetos detectados", len(detections))
m2.metric("Tiempo de inferencia", f"{result['inference_ms']:.0f} ms")
precision = result["precision"].upper()
m3.metric("Dispositivo", f"GPU (CUDA) · {precision}" if result["device"] == "cuda" else f"CPU · {precision}")
m4.metric("Resolución", f"{result['image_width']}×{result['image_height']}")

# ---------- Tabla ----------
st.subheader("Detalle")
if detections:
    table = pd.DataFrame(
        {
            "Objeto": [d["label"] for d in detections],
            "Confianza": [d["confidence"] * 100 for d in detections],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Confianza": st.column_config.ProgressColumn(
                "Confianza", min_value=0, max_value=100, format="%.1f%%"
            )
        },
    )
else:
    st.warning("No se detectó ningún objeto con este umbral. Prueba a bajarlo en la barra lateral.")
