FROM python:3.10-slim

# System dependencies for OpenCV headless, ONNX runtime, and FFmpeg (RTSP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /safevision

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY app/ ./app/
COPY models/ ./models/

# Headless mode — no GUI on a server
ENV DISPLAY_OUTPUT=false
# Ensure Python output is not buffered (important for Docker logs)
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Health check for Docker / GCE
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "app.main"]
