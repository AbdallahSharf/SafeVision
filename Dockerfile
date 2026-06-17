# Use the official, ultra-lightweight Python slim image
FROM python:3.11-slim

# System dependencies for OpenCV headless, FFmpeg (RTSP), GStreamer (GPU decode), and ONNXRuntime/TensorRT
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libgomp1 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/safevision

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torch torchvision onnxruntime && \
    pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir onnxruntime-gpu==1.19.2

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Copy local models directly
COPY models/ ./models/

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
