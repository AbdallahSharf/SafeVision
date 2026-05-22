# Use NVIDIA CUDA 12.1 runtime on Ubuntu 22.04 — enables GPU inside the container
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies for Python 3.10, OpenCV headless, FFmpeg (RTSP),
# and build tools required to compile insightface's Cython extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /safevision

# Install Python dependencies (cached layer — reinstall only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Health check — give extra time for GPU models to load on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "app.main"]
