# 🛡️ SafeVision

**Real-time AI-powered face recognition security system.**

SafeVision processes a live RTSP camera feed, detects faces using a custom YOLO model, recognizes identities via ArcFace embeddings matched against a MongoDB Atlas vector database, and streams the annotated video to mobile applications in real time.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![GCP](https://img.shields.io/badge/Google%20Cloud-GCE-4285F4?logo=googlecloud)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

```
┌──────────────┐    RTSP     ┌─────────────────────────────────────┐
│  IP Camera   │────────────▶│         SafeVision Server           │
│  (DDNS URL)  │             │                                     │
└──────────────┘             │  ┌───────────┐   ┌──────────────┐  │
                             │  │ YOLO Face │──▶│  ArcFace     │  │
                             │  │ Detector  │   │  Embedder    │  │
                             │  └───────────┘   └──────┬───────┘  │
                             │                         │          │
                             │  ┌──────────────────────▼───────┐  │
                             │  │ MongoDB Atlas Vector Search  │  │
                             │  └──────────────────────────────┘  │
                             │                                     │
                             │  ┌──────────────────────────────┐  │
                             │  │  FastAPI (MJPEG + REST API)  │  │
                             │  └──────────────┬───────────────┘  │
                             └─────────────────┼──────────────────┘
                                               │
                            ┌──────────────────┼──────────────────┐
                            │                  │                  │
                       GET /stream        GET /faces        GET /status
                            │                  │                  │
                       ┌────▼────┐       ┌─────▼─────┐     ┌─────▼─────┐
                       │ Mobile  │       │  Mobile   │     │ Monitor   │
                       │  App    │       │  App      │     │ Dashboard │
                       │ (Video) │       │ (Data)    │     │           │
                       └─────────┘       └───────────┘     └───────────┘
```

## ✨ Features

- **Real-time face detection** — Custom YOLO model optimized for faces
- **Face recognition** — ArcFace (InsightFace) with MongoDB Atlas vector search
- **Live video streaming** — MJPEG endpoint consumable by mobile apps and browsers
- **Low-light enhancement** — Automatic CLAHE + gamma correction + denoising
- **Temporal smoothing** — Reduces flickering identity labels across frames
- **REST API** — Health checks, status metrics, and recent faces endpoint
- **Docker-ready** — Single command to build and deploy
- **CI/CD** — GitHub Actions auto-deploys to Google Compute Engine

---

## 📁 Project Structure

```
SafeVision/
├── app/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Centralized config (env vars + dotenv)
│   ├── database.py          # MongoDB Atlas connection
│   ├── models_loader.py     # Lazy YOLO + ArcFace loading
│   ├── enhancement.py       # Low-light image enhancement
│   ├── recognition.py       # Face recognition (vector search)
│   ├── stream.py            # Threaded RTSP video reader
│   ├── processor.py         # Frame processing pipeline
│   ├── api.py               # FastAPI server (MJPEG + REST)
│   └── main.py              # Entry point
├── models/
│   ├── best.pt              # YOLO face detector weights (Git LFS)
│   └── w600k_r50.onnx       # ArcFace embedding model (Git LFS)
├── scripts/
│   └── deploy.sh            # Manual GCE deployment script
├── .github/workflows/
│   └── deploy.yml           # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── .env.example             # Environment variable template
├── requirements.txt         # Pinned Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10**
- **MongoDB Atlas** cluster with a vector search index named `vector_index` on the `faces` collection
- **RTSP camera** accessible via network (local or DDNS)
- **Git LFS** installed (for model files)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SafeVision.git
cd SafeVision
git lfs pull  # Download model weights
```

### 2. Set up environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your real MONGO_URI and RTSP_URL
```

### 3. Run locally

```bash
python -m app.main
```

The server starts at `http://localhost:8080`. Open `http://localhost:8080/stream` in a browser to see the live annotated feed.

---

## 🐳 Docker

### Build and run

```bash
docker-compose up --build
```

### Or manually

```bash
docker build -t safevision .
docker run -d --name safevision -p 8080:8080 --env-file .env safevision
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check (200 if healthy, 503 if degraded) |
| `/status` | GET | System metrics: FPS, stream state, DB face count |
| `/stream` | GET | Live MJPEG video stream with face recognition overlays |
| `/faces` | GET | Recently recognized faces (JSON) |

### `/stream` — Live Video

Open in a browser or use in your mobile app:

```
http://<server-ip>:8080/stream
```

This returns a continuous MJPEG stream. In mobile apps, load it in an image/video view component.

### `/faces` — Recent Detections

```bash
curl http://localhost:8080/faces?limit=5
```

```json
{
  "faces": [
    {
      "name": "Ahmad",
      "confidence": 0.91,
      "bbox": [120, 45, 230, 190],
      "timestamp": 1716321234.56
    }
  ]
}
```

### `/status` — System Status

```bash
curl http://localhost:8080/status
```

```json
{
  "stream_connected": true,
  "fps": 14.2,
  "faces_in_db": 150,
  "uptime_seconds": 3600.5,
  "config": {
    "frame_size": "800x600",
    "yolo_conf": 0.4,
    "recog_threshold": 0.6,
    "low_light_enabled": true
  }
}
```

---

## ☁️ Google Compute Engine Deployment

### 1. Create a GCE VM

```bash
gcloud compute instances create safevision-vm \
    --zone=me-west1-a \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --tags=http-server
```

### 2. Allow HTTP traffic

```bash
gcloud compute firewall-rules create allow-safevision \
    --allow=tcp:8080 \
    --target-tags=http-server \
    --description="Allow SafeVision API access"
```

### 3. Set up secrets on the VM

```bash
# SSH into the VM
gcloud compute ssh safevision-vm --zone=me-west1-a

# Create env file
sudo mkdir -p /opt/safevision
sudo nano /opt/safevision/.env
# Paste your MONGO_URI and RTSP_URL (with DDNS hostname)
```

### 4. Deploy

```bash
# From your local machine:
./scripts/deploy.sh
```

### 5. Access the stream

```
http://<VM-EXTERNAL-IP>:8080/stream
```

---

## 🔄 CI/CD (GitHub Actions)

Auto-deploys to GCE on every push to `main`.

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_SA_KEY` | Service account JSON key (base64-encoded) |
| `GCE_VM_NAME` | Compute Engine VM name |
| `GCE_VM_ZONE` | VM zone (e.g., `me-west1-a`) |
| `GCP_REGION` | Artifact Registry region (e.g., `me-west1`) |

---

## 🌐 DDNS Setup (Camera Access)

Your RTSP camera needs to be accessible from the GCE VM. With DDNS:

1. **Set up DDNS** on your router (e.g., No-IP, DuckDNS, or your router's built-in DDNS)
2. **Port forward** port `554` (RTSP) on your router to the camera's local IP (`192.168.1.9`)
3. **Update** `RTSP_URL` in your `.env`:
   ```
   RTSP_URL=rtsp://admin:password@your-hostname.ddns.net:554/cam/realmonitor?channel=1&subtype=0
   ```

---

## ⚙️ Environment Variables

See [`.env.example`](.env.example) for all available variables with descriptions.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGO_URI` | ✅ | — | MongoDB Atlas connection string |
| `RTSP_URL` | ✅ | — | RTSP camera URL |
| `PORT` | ❌ | `8080` | API server port |
| `DISPLAY_OUTPUT` | ❌ | `false` | Enable OpenCV GUI window |
| `YOLO_CONF_THRESHOLD` | ❌ | `0.4` | YOLO confidence threshold |
| `BOX_CONF_THRESHOLD` | ❌ | `0.6` | Bounding box confidence |
| `RECOG_THRESHOLD` | ❌ | `0.6` | Face recognition threshold |
| `LOW_LIGHT_ENABLE` | ❌ | `true` | Auto low-light enhancement |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot open RTSP stream` | Check camera URL, DDNS, port forwarding |
| `Cannot connect to MongoDB` | Verify `MONGO_URI`, check Atlas network access (whitelist VM IP) |
| Models load slowly | First run downloads/loads ~170 MB of weights — subsequent starts are faster |
| Low FPS | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT`, set `DENOISE_STRENGTH=0` |
| `cv2.imshow` crash on server | Ensure `DISPLAY_OUTPUT=false` (default in Docker) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
