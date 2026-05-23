# GPU-enabled base image with PyTorch and CUDA pre-installed
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# System dependencies for OpenCV headless and FFmpeg (RTSP decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /safevision

# Install Python dependencies
# Strip out CPU torch/onnxruntime from requirements so we keep the CUDA versions
COPY requirements.txt .
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/onnxruntime/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime-gpu

# Copy application code and model weights
COPY app/ ./app/
COPY models/ ./models/

# Headless mode — no GUI on a server
ENV DISPLAY_OUTPUT=false
# Ensure Python output is not buffered (important for Docker logs)
ENV PYTHONUNBUFFERED=1
# Silence noisy FFmpeg HEVC warnings
ENV OPENCV_FFMPEG_LOGLEVEL=16

EXPOSE 8080

# Health check — give models time to load on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "app.main"]
