# Use the official, ultra-lightweight Python slim image
FROM python:3.11-slim

# System dependencies for OpenCV headless, FFmpeg (RTSP), and ONNXRuntime/TensorRT
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/safevision

COPY requirements.txt .
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/onnxruntime/d' requirements.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime-gpu

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Download model weights from Google Cloud Storage at build time.
# Models are stored in GCS instead of Git LFS to avoid LFS bandwidth quota limits.
# GCS_MODELS_BUCKET build-arg must be set (e.g. gs://safevision-models/models).
ARG GCS_MODELS_BUCKET
RUN if [ -z "${GCS_MODELS_BUCKET}" ]; then \
      echo "ERROR: GCS_MODELS_BUCKET build-arg is required (e.g. gs://my-bucket/models)" && exit 1; \
    fi && \
    pip install --no-cache-dir gsutil && \
    mkdir -p models && \
    gsutil -m cp "${GCS_MODELS_BUCKET}/best.pt" models/best.pt && \
    gsutil -m cp "${GCS_MODELS_BUCKET}/w600k_r50.onnx" models/w600k_r50.onnx && \
    echo "Models downloaded successfully." && \
    pip uninstall -y gsutil

# Headless mode — no GUI on a server
ENV DISPLAY_OUTPUT=false
# Ensure Python output is not buffered (important for Docker logs)
ENV PYTHONUNBUFFERED=1
# Silence noisy FFmpeg HEVC warnings
ENV OPENCV_FFMPEG_LOGLEVEL=16

# Expose CUDA libraries installed via pip to ONNX Runtime and TensorRT
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/tensorrt_libs:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

EXPOSE 8080

# Health check — give models time to load on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "app.main"]
