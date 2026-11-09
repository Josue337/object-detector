# Detector de objetos con IA

Aplicación web que detecta objetos en imágenes usando un modelo de Hugging Face (DETR ResNet-50) acelerado con CUDA. El usuario sube una imagen o toma una foto con la cámara, y la interfaz muestra los objetos detectados con sus cajas, su porcentaje de confianza, el tiempo de inferencia y el dispositivo utilizado (GPU o CPU).

## Autores

- Josue Santiago Bustamante Luna
- Juan David Guzman Perez

## Características

- Subida de imágenes (JPEG, PNG, WebP) o captura desde la cámara del navegador.
- Imagen original y imagen con las cajas delimitadoras, lado a lado.
- Tabla con cada objeto y su porcentaje de confianza.
- Umbral de confianza ajustable desde la interfaz.
- Indicador de dispositivo (GPU CUDA o CPU), precisión (FP16 o FP32) y tiempo de inferencia.
- API REST documentada automáticamente en `/docs`.
- Inferencia en FP16 sobre GPU, con calentamiento del modelo al arrancar.
- Validación de entradas y manejo de errores con códigos HTTP específicos.

## Arquitectura

```
Streamlit (frontend)  ──HTTP POST /detect──▶  FastAPI (backend)
  sube la imagen                                valida la imagen
  dibuja cajas y métricas                       preprocesa (Transformers)
                                                inferencia (PyTorch + CUDA)
                      ◀────── JSON ───────      postprocesa y responde
```

| Capa | Archivo | Responsabilidad |
|---|---|---|
| Interfaz | `frontend/app.py` | Subida de imagen, llamada a la API, dibujo de cajas con OpenCV y métricas |
| API | `backend/main.py` | Rutas HTTP, validación de entradas, manejo de errores |
| Lógica | `backend/detector.py` | Inferencia: imagen entrada, lista de detecciones salida |
| Modelo | `backend/model.py` | Selección de dispositivo y carga del modelo |

El modelo se carga **una sola vez** al iniciar el servidor, y las capas están separadas para poder cambiar el modelo sin tocar la API ni la interfaz.

## Tecnologías

Python · PyTorch · Hugging Face Transformers · timm · FastAPI · Uvicorn · Streamlit · OpenCV · Pillow · CUDA

## Requisitos

- Linux o WSL2 (probado en Ubuntu 26.04 sobre WSL2) con Python 3.10 o superior (probado con 3.14).
- GPU NVIDIA con driver actualizado (recomendado). Sin GPU el proyecto funciona en CPU, pero más lento.
- Si usas WSL2, el driver de NVIDIA se instala **solo en Windows**; no instales ningún driver dentro de Ubuntu.

## Instalación

```bash
git clone https://github.com/Josue337/object-detector.git
cd object_detector

python3 -m venv .venv
source .venv/bin/activate
```

Instala PyTorch con el comando que indica <https://pytorch.org/get-started/locally/> para tu sistema y tu versión de CUDA. Tiene esta forma:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

> **Instala PyTorch antes que el resto.** Si no, `timm` lo trae como dependencia y pip descarga la versión por defecto, con varios GB de librerías CUDA que pueden no coincidir con tu configuración. Tampoco está en `requirements.txt`, porque el comando depende de la versión de CUDA de cada máquina.

Después, instala el resto de dependencias:

```bash
pip install -r backend/requirements.txt -r frontend/requirements.txt
```

Verifica que PyTorch detecta la GPU:

```bash
python check_cuda.py
```

La primera ejecución del backend descarga el modelo (unos 160 MB) en la carpeta `models/`.

## Uso

Necesitas dos terminales, ambas con el entorno virtual activado.

**Terminal 1: backend**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2: frontend**

```bash
cd frontend
streamlit run app.py
```

Abre <http://localhost:8501> en el navegador. La documentación interactiva de la API está en <http://localhost:8000/docs>.

> Los comandos se ejecutan desde cada carpeta (`backend/` y `frontend/`) porque los imports y la configuración de Streamlit son relativos a ellas.

## API

### `GET /health`

Estado del servidor.

```json
{"status": "ok", "device": "cuda", "precision": "fp16"}
```

### `POST /detect`

Recibe una imagen (`multipart/form-data`, campo `file`) y devuelve los objetos detectados.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `file` | archivo | Imagen JPEG, PNG o WebP (máximo 10 MB y 50 megapíxeles) |
| `threshold` | query, 0 a 1 | Confianza mínima. Por defecto `0.9` |

```bash
curl -X POST "http://localhost:8000/detect?threshold=0.9" -F "file=@test.jpg"
```

```json
{
  "device": "cuda",
  "precision": "fp16",
  "inference_ms": 30.2,
  "image_width": 640,
  "image_height": 480,
  "detections": [
    {"label": "cat", "confidence": 0.9988, "box": [13.2, 52.0, 314.0, 470.9]},
    {"label": "remote", "confidence": 0.9982, "box": [40.2, 70.8, 175.6, 118.0]}
  ]
}
```

Las cajas están en píxeles con formato `[x_min, y_min, x_max, y_max]`.

### Códigos de error

| Código | Causa |
|---|---|
| 400 | El archivo no es una imagen válida |
| 413 | La imagen supera los 10 MB o los 50 megapíxeles |
| 415 | Formato no soportado |
| 422 | Parámetro inválido (por ejemplo, `threshold` fuera de 0 a 1) |
| 500 | Error interno durante la inferencia |
| 503 | La GPU se quedó sin memoria |

## Configuración

| Variable | Por defecto | Descripción |
|---|---|---|
| `MODEL_NAME` | `facebook/detr-resnet-50` | Modelo de Hugging Face para detección de objetos |
| `USE_FP16` | `1` | Usa media precisión en GPU (`0` para desactivarla) |
| `API_URL` | `http://localhost:8000` | Dirección del backend, leída por el frontend |

Ejemplo: `USE_FP16=0 uvicorn main:app --port 8000`

## Rendimiento

Medido en una NVIDIA GeForce RTX 4050 Laptop (6 GB), con una imagen de 640×480, como media de 30 ejecuciones. El tiempo es el de la red en GPU y no incluye el pre y postprocesamiento, que se hacen en CPU.

| Dispositivo y precisión | Tiempo medio | VRAM pico |
|---|---|---|
| CPU (FP32) | 997.5 ms | no aplica |
| GPU CUDA (FP32) | 57.0 ms | 401 MB |
| **GPU CUDA (FP16)** | **30.7 ms** | **203 MB** |

Con FP16 las detecciones y sus confianzas coinciden con las de FP32 hasta el cuarto decimal. La primera petición tras arrancar el servidor es algo más lenta (~250 ms) por la inicialización de recursos de CUDA.

## Estructura del proyecto

```
object_detector/
├── backend/
│   ├── main.py              # API FastAPI
│   ├── detector.py          # Lógica de inferencia
│   ├── model.py             # Carga del modelo y dispositivo
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Interfaz Streamlit
│   └── requirements.txt
├── models/                  # Caché de pesos (no se versiona)
├── check_cuda.py            # Verificación de PyTorch y CUDA
├── test_model.py            # Prueba mínima del modelo
└── README.md
```

## Decisiones de diseño

- **El frontend dibuja las cajas y la API devuelve JSON.** La API queda reutilizable por otros clientes y la presentación se cambia sin tocar el modelo.
- **Modelo cargado al arrancar** con el `lifespan` de FastAPI, y no en cada petición.
- **Endpoint síncrono (`def`).** La inferencia bloquea; FastAPI la ejecuta en un pool de hilos y el servidor sigue atendiendo peticiones.
- **Un `Lock` serializa el acceso a la GPU**, para que varias peticiones simultáneas no disparen el consumo de memoria.
- **Validación en capas:** tipo declarado, tamaño en bytes, decodificación real de la imagen y número de píxeles, con un código HTTP distinto para cada fallo.
- **Postprocesamiento en FP32** aunque la red corra en FP16, para no perder precisión en las coordenadas.

## Limitaciones y trabajo futuro

- Empaquetado con Docker y soporte de GPU en contenedores.
- Pruebas automatizadas de la API con `pytest`.
- Autenticación y límite de peticiones para uso público.
- Procesamiento de vídeo en tiempo real.
- Comparar con otros modelos (YOLOS, RT-DETR).
## Licencia

Este proyecto se distribuye bajo la licencia MIT (ver `LICENSE`). El modelo DETR ResNet-50 es de AI at Meta y tiene su propia licencia, Apache 2.0.

## Autores

- Josue Santiago Bustamante Luna
- Juan David Guzman Perez
