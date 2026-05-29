# 🛡️ SafeVision

**Real-time AI-powered face recognition security system.**

SafeVision processes a live RTSP camera feed, detects faces using a custom YOLO model, recognizes identities via ArcFace embeddings matched against a MongoDB Atlas vector database, and streams the annotated video in real time over a pure MJPEG endpoint.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![GCP](https://img.shields.io/badge/Google%20Cloud-GPU%20VM-4285F4?logo=googlecloud)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Tesla%20T4-76B900?logo=nvidia)
![Tailscale](https://img.shields.io/badge/Tailscale-VPN-0e8a8a?logo=tailscale)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

```
┌──────────────┐  RTSP (TCP)  ┌──────────────────────────────────────────────────────────┐
│  IP Camera   │─────────────▶│                  SafeVision Server                       │
│ 192.168.1.x  │  Tailscale   │                                                          │
└──────────────┘    VPN       │   Thread 1 — RTSP Reader                                 │
                              │   ┌──────────────────────────────────────────────┐        │
                              │   │  VideoStream.read_latest() → raw BGR frame   │        │
                              │   └────────────────────┬─────────────────────────┘        │
                              │                        │ _rtsp_queue (maxsize=2)           │
                              │   Thread 2 — YOLO Detector                               │
                              │   ┌────────────────────▼─────────────────────────┐        │
                              │   │  YOLO(best.pt / best.onnx) → bounding boxes  │        │
                              │   └────────────────────┬─────────────────────────┘        │
                              │                        │ _detect_queue (maxsize=2)         │
                              │   Thread 3 — ArcFace Recognizer                          │
                              │   ┌────────────────────▼─────────────────────────┐        │
                              │   │  ArcFace ONNX → embedding → MongoDB search   │        │
                              │   └────────────────────┬─────────────────────────┘        │
                              │                        │ _latest_ai_results               │
                              │   MJPEG Generator (decoupled from AI pipeline)           │
                              │   ┌────────────────────▼─────────────────────────┐        │
                              │   │  raw frame + overlay latest AI boxes → JPEG  │        │
                              │   └────────────────────┬─────────────────────────┘        │
                              └────────────────────────┼─────────────────────────────────┘
                                                       │
                               ┌───────────────────────┼───────────────────────┐
                               │                       │                       │
                          GET /stream            GET /faces               GET /status
                               │                       │                       │
                          ┌────▼────┐           ┌──────▼──────┐         ┌──────▼──────┐
                          │ Mobile  │           │   Mobile    │         │  Monitoring │
                          │  App    │           │  App (Data) │         │  Dashboard  │
                          │ (Video) │           └─────────────┘         └─────────────┘
                          └─────────┘
```

---

## ✨ Features

- **Pure MJPEG streaming** — Ultra-low-latency stream consumable by any browser or mobile app
- **Decoupled AI pipeline** — MJPEG generator runs independently from YOLO + ArcFace, guaranteeing smooth video even when AI inference lags
- **Custom YOLO face detector** — Fine-tuned on face data for high-precision detection
- **ArcFace recognition** — 512-d embeddings via ONNX Runtime (CUDA-accelerated), matched against MongoDB Atlas vector search
- **ByteTrack-style IoU tracking** — Each face gets a persistent integer ID across frames, eliminating identity flicker
- **Adaptive recognition thresholds** — Per-identity calibrated thresholds based on enrollment embedding variance
- **Blur quality gate** — Skips recognition on blurry/motion-blurred face crops to avoid polluting identity history
- **Batch ArcFace inference** — Multiple visible faces processed in a single ONNX batched call
- **Low-light enhancement** — CLAHE + gamma correction + optional denoising
- **Firebase push alerts** — Real-time push notification fires when an unauthorized face is detected
- **Offline camera resilience** — Server stays online and shows a placeholder frame if the camera disconnects
- **REST API** — Health checks, status metrics, recent faces, and admin endpoints
- **Docker-ready** — Single command to build and deploy
- **GPU-accelerated** — Runs on an NVIDIA Tesla T4 on Google Cloud (me-west1-b)
- **CI/CD** — GitHub Actions auto-builds and pushes Docker images on every push to `main`

---

## 🧰 Tech Stack & Why We Chose It

### 🐍 Python 3.11
The de facto language for AI/ML. Its ecosystem — PyTorch, OpenCV, ONNX Runtime — made it the only practical choice. Version 3.11 gives a measurable performance uplift over 3.10 due to the Faster CPython project.

### ⚡ FastAPI
FastAPI serves dual duty: it streams MJPEG video frames and exposes REST endpoints. We chose it over Flask because:
- **Async by default** — Non-blocking request handling is critical for a streaming server.
- **Automatic OpenAPI docs** — `/docs` is available out-of-the-box for free.
- **Pydantic validation** — Strict typing catches bugs early.

### 🤖 YOLOv11 (Ultralytics) — Face Detection
YOLO (You Only Look Once) is a single-shot object detector that runs in real time. We use a **custom-trained** YOLO model (`best.pt`) specifically fine-tuned on face data. It runs on CUDA via PyTorch and produces an ONNX export for TensorRT compatibility.

### 🧠 ArcFace (w600k_r50.onnx) — Face Recognition
ArcFace is state-of-the-art face recognition producing 512-dimensional embeddings. We load the pre-trained `w600k_r50.onnx` (ResNet-50 backbone, trained on 600k identities) **directly via ONNX Runtime**, eliminating the `insightface` Cython dependency entirely. ONNX Runtime runs transparently on CUDAExecutionProvider for GPU acceleration.

### 🍃 MongoDB Atlas — Vector Database
After ArcFace produces an embedding, we find the closest match using MongoDB Atlas's **$vectorSearch** (ANN search) — no separate vector database like Pinecone is needed.

### 🐳 Docker
The entire application is containerized. The base image is `nvidia/cuda:12.2.0-runtime-ubuntu22.04` to support GPU inference. The image is stored in **Google Artifact Registry** (`me-west1-docker.pkg.dev`).

### ☁️ Google Compute Engine (GCE) — me-west1-b (Tel Aviv)
The cloud VM that runs the SafeVision Docker container. A GPU VM in **Tel Aviv** (`me-west1-b`) was chosen to minimize latency to the camera and end users. The VM uses an **NVIDIA Tesla T4** for GPU inference.

### 🔄 GitHub Actions — CI/CD
On every push to `main`, the pipeline automatically:
1. Checks out code **including Git LFS model files**.
2. Builds a new Docker image.
3. Pushes it to Google Artifact Registry.
4. SSHs into the GCE VM and hot-swaps the container with zero downtime.

Authentication uses **Workload Identity Federation** (keyless auth — no long-lived JSON keys stored as secrets).

### 🔒 Tailscale — Secure Camera VPN
The IP camera sits on a private home network. The GCE VM is on Google Cloud. Tailscale (WireGuard-based zero-config VPN) connects them securely. The local Windows PC acts as a **subnet router**, advertising `192.168.1.0/24` to the Tailscale mesh so the GCE VM can reach the camera directly.

---

## 📁 Project Structure

```
SafeVision/
├── app/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Centralized config (env vars + dotenv)
│   ├── database.py          # MongoDB Atlas connection (sync + async)
│   ├── models_loader.py     # Lazy YOLO + ArcFace ONNX loading (thread-safe singletons)
│   ├── enhancement.py       # Low-light image enhancement (CLAHE + gamma + denoising)
│   ├── recognition.py       # Face recognition (MongoDB vector search + adaptive thresholds)
│   ├── tracker.py           # ByteTrack-style IoU multi-object tracker
│   ├── stream.py            # Threaded RTSP video reader (always-latest frame)
│   ├── processor.py         # 2-stage frame processor (detect → recognize_faces)
│   ├── alerts.py            # Firebase Cloud Messaging push alerts
│   ├── api.py               # FastAPI server — 3-thread pipeline + MJPEG + REST
│   └── main.py              # Entry point (TensorRT engine build + uvicorn)
├── models/
│   ├── best.pt              # YOLO face detector weights (Git LFS)
│   └── w600k_r50.onnx       # ArcFace embedding model (Git LFS)
├── scripts/
│   ├── enroll_face.py       # CLI to enroll face photos into MongoDB
│   └── export_tensorrt.py   # Export YOLO to TensorRT .engine format
├── .github/workflows/
│   └── deploy.yml           # CI/CD pipeline (build → push → deploy)
├── Dockerfile
├── docker-compose.yml
├── .env.example             # Environment variable template
├── requirements.txt         # Pinned Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11**
- **MongoDB Atlas** cluster with a vector search index named `vector_index` on the `faces` collection
- **RTSP camera** accessible via network (local or Tailscale VPN)
- **Git LFS** installed (for model files)

### 1. Clone the repository

```bash
git clone https://github.com/AbdallahSharf/SafeVision.git
cd SafeVision
git lfs pull  # Download model weights (~175 MB)
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

The server starts at `http://localhost:8080`. Open `http://localhost:8080/stream` in a browser to see the live annotated MJPEG feed.

---

## 🐳 Docker

### Build and run

```bash
docker-compose up --build
```

### GPU deployment (recommended)

```bash
docker build -t safevision .
docker run -d \
  --name safevision \
  --restart unless-stopped \
  --network host \
  --gpus all \
  -v /opt/safevision/unauthorized_faces:/opt/safevision/unauthorized_faces \
  --env-file /opt/safevision/.env \
  safevision
```

---

## 📡 API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | — | API info and available endpoints |
| `/health` | GET | — | Health check (200 healthy / 503 degraded) |
| `/status` | GET | — | System metrics: FPS, pipeline state, DB face count |
| `/stream` | GET | — | Live MJPEG video stream with face recognition overlays |
| `/faces` | GET | Bearer | Recently recognized faces (JSON) |
| `/alerts` | GET | Bearer | Recent unauthorized access alerts |
| `/images/{filename}` | GET | — | Saved unauthorized face images |
| `/admin/reload-thresholds` | POST | Bearer | Recompute per-identity recognition thresholds |

### `/stream` — Live MJPEG Video

Open in a browser or mobile app:

```
http://<server-ip>:8080/stream
```

### `/faces` — Recent Detections

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8080/faces?limit=5
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
  "pipeline": {
    "reader": "running",
    "detector": "running",
    "recognizer": "running"
  },
  "fps": 15.2,
  "faces_in_db": 12,
  "uptime_seconds": 3600.5,
  "config": {
    "frame_size": "1280x720",
    "jpeg_quality": 65,
    "yolo_conf": 0.4,
    "recog_threshold": 0.6,
    "low_light_enabled": true
  }
}
```

---

## ☁️ Google Compute Engine Deployment

### 1. Create a GPU VM in Tel Aviv

```bash
gcloud compute instances create safevision-gpu-vm \
    --zone=me-west1-b \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --tags=safevision-server \
    --metadata=install-nvidia-driver=True
```

### 2. Allow HTTP traffic on port 8080

```bash
gcloud compute firewall-rules create allow-safevision \
    --allow=tcp:8080 \
    --target-tags=safevision-server \
    --description="Allow SafeVision API access"
```

### 3. Create the `.env` file on the VM

```bash
gcloud compute ssh safevision-gpu-vm --zone=me-west1-b

sudo mkdir -p /opt/safevision/unauthorized_faces
sudo nano /opt/safevision/.env
# Paste your env vars, then save
```

### 4. Set up Tailscale VPN (for local camera access)

See the [Tailscale Setup](#-tailscale-vpn-setup) section below.

### 5. Deploy via GitHub Actions

Push to `main` — the CI/CD pipeline handles everything else automatically.

> **Note:** Make sure your `GCE_VM_ZONE` GitHub secret is set to `me-west1-b`.

### 6. Access the stream

```
http://<VM-EXTERNAL-IP>:8080/stream
```

---

## 🔄 CI/CD (GitHub Actions)

Auto-builds and pushes on every commit to `main`. Uses **Workload Identity Federation** (no JSON keys required).

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID (e.g. `safevision-00`) |
| `GCP_WIF_PROVIDER` | Workload Identity Provider resource name |
| `GCP_SA_EMAIL` | Service account email |
| `GCE_VM_NAME` | Compute Engine VM name (e.g. `safevision-gpu-vm`) |
| `GCE_VM_ZONE` | VM zone — must be `me-west1-b` |
| `GCE_SSH_PRIVATE_KEY` | Private SSH key for connecting to the VM |

> **Note:** The `.env` file (containing `MONGO_URI`, `RTSP_URL`, etc.) lives directly on the VM at `/opt/safevision/.env` and is **not** stored in GitHub. The deployment pipeline mounts it into the container at runtime.

> **Important:** Model files (`best.pt`, `w600k_r50.onnx`) are stored in Git LFS. The CI/CD workflow uses `actions/checkout@v4` with `lfs: true` to ensure the real model weights (not LFS pointer files) are included in the Docker image.

---

## 🔒 Tailscale VPN Setup

Tailscale creates a secure WireGuard VPN between your GCE VM and your local network, allowing the VM to reach your IP camera directly by local IP — no port forwarding or DDNS required.

### 1. Install Tailscale on your Windows PC

Download and install from [tailscale.com/download](https://tailscale.com/download) and log in with your account.

### 2. Advertise your camera subnet

Open **Command Prompt as Administrator** and run:

```cmd
tailscale up --advertise-routes=192.168.1.0/24
```

Then go to the [Tailscale Admin Console](https://login.tailscale.com/admin/machines), find your PC, click **Edit route settings**, and approve the advertised route.

### 3. Install Tailscale on the VM

```bash
gcloud compute ssh safevision-gpu-vm --zone=me-west1-b --command="curl -fsSL https://tailscale.com/install.sh | sudo sh && sudo tailscale up --accept-routes"
```

Open the auth URL it prints and log in with the **same account** as Step 1.

### 4. Update RTSP URL

Once connected, set your VM's `/opt/safevision/.env`:

```env
RTSP_URL=rtsp://admin:your_password@192.168.1.x:554/cam/realmonitor?channel=1&subtype=0
```

---

## ⚙️ Environment Variables

See [`.env.example`](.env.example) for all variables with descriptions.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGO_URI` | ✅ | — | MongoDB Atlas connection string |
| `RTSP_URL` | ✅ | — | RTSP camera URL |
| `PORT` | ❌ | `8080` | API server port |
| `YOLO_MODEL_PATH` | ❌ | `models/best.pt` | Path to YOLO weights |
| `ARCFACE_MODEL_PATH` | ❌ | `models/w600k_r50.onnx` | Path to ArcFace ONNX model |
| `YOLO_CONF_THRESHOLD` | ❌ | `0.4` | YOLO detection confidence threshold |
| `BOX_CONF_THRESHOLD` | ❌ | `0.6` | Bounding box confidence filter |
| `RECOG_THRESHOLD` | ❌ | `0.6` | Face recognition match threshold |
| `BLUR_THRESHOLD` | ❌ | `80.0` | Laplacian variance below which face crops are skipped |
| `LOW_LIGHT_ENABLE` | ❌ | `true` | Auto low-light enhancement (CLAHE) |
| `FRAME_WIDTH` | ❌ | `1280` | Processing resolution width |
| `FRAME_HEIGHT` | ❌ | `720` | Processing resolution height |
| `STREAM_JPEG_QUALITY` | ❌ | `65` | MJPEG stream JPEG quality (0–100) |
| `FACE_SIZE` | ❌ | `112` | ArcFace input face crop size (px) |
| `FACE_MARGIN` | ❌ | `20` | Bounding box expansion margin (px) |
| `API_SECRET_KEY` | ❌ | — | Bearer token for protected endpoints |
| `FIREBASE_CREDENTIALS` | ❌ | — | Path to Firebase service account JSON |

---

## 🚀 What We Built — Full Development History

### v1.0 — Foundation
- FastAPI server with MJPEG `/stream` endpoint
- YOLO face detection + InsightFace ArcFace recognition
- MongoDB Atlas vector search for identity lookup
- Basic Tailscale VPN setup for camera connectivity
- Dockerized deployment to GCE

### v1.5 — Stability & Reliability
- Fixed server crash when camera was offline (replaced hard `RuntimeError` with graceful retry)
- Added "Camera Connecting or Offline" placeholder frame so browser never hangs
- Forced RTSP TCP transport to eliminate UDP packet loss over Tailscale VPN
- Added `set -e` to deployment script to surface silent failures
- Added `/health` endpoint for load balancer probes

### v2.0 — Performance Architecture Overhaul
- **3-thread pipeline**: Split the monolithic processing loop into three concurrent threads (Reader → YOLO Detector → ArcFace Recognizer) connected by bounded `queue.Queue(maxsize=2)`. Frames are never accumulated — stale frames are dropped to keep latency minimal.
- **Decoupled MJPEG generator**: The video stream now runs completely independently of the AI pipeline. It pulls the freshest raw camera frame and overlays the *latest known* AI bounding boxes. This guarantees smooth video at camera FPS even when AI inference is slower.
- **IoU-based ByteTrack-style tracker**: Replaced grid-based smoothing with a proper multi-object tracker (`tracker.py`). Each face gets a persistent integer ID, enabling correct handling of multiple people and eliminating identity flicker.
- **Adaptive recognition thresholds**: Per-identity calibrated thresholds based on the standard deviation of enrolled embeddings. Consistent enrollees get tight thresholds; varied enrollees get looser ones.
- **Blur quality gate**: Laplacian variance computed on face crops every 5 frames (cached). Blurry crops are skipped entirely.
- **Batch ArcFace inference**: Up to 4 face crops batched into a single ONNX Runtime call for proportional throughput improvement on GPU.
- **Async DB background thread**: MongoDB vector search runs in a dedicated asyncio event loop thread so it never blocks YOLO or the MJPEG generator.
- **Multi-angle face enrollment**: Added `scripts/enroll_face.py` for bulk enrollment of multiple photos per identity.

### v2.5 — Security & Alerts
- **API authentication**: HTTP Bearer Token checks (FastAPI `Depends`) on `/faces`, `/alerts`, and `/admin` endpoints.
- **Firebase push alerts**: Real-time FCM push notification fires on first unauthorized face detection, including confidence score and saved face image.
- **Unauthorized face logging**: Saves a cropped JPEG of every unauthorized face to `/opt/safevision/unauthorized_faces/` and serves it via `/images/` static endpoint.
- **`/alerts` endpoint**: Mobile app can query the full alert history from MongoDB.
- **`/admin/reload-thresholds`**: Hot-reload calibrated thresholds after enrolling new faces — no restart required.

### v3.0 — GPU Migration & MJPEG Optimization
- **Migrated to NVIDIA Tesla T4 GPU** on Google Cloud `me-west1-b` (Tel Aviv zone) for lower latency.
- **Removed WebRTC entirely**: Stripped `aiortc` and all WebRTC code. Pure MJPEG `/stream` is the single video endpoint — simpler, faster, and universally compatible.
- **Removed `aiortc` and `insightface`**: ArcFace now loaded directly via ONNX Runtime — no Cython compilation required. Eliminates fragile C++ build dependencies from the Docker image.
- **Removed `DETECT_EVERY_N` frame-skipping**: YOLO now runs on every single frame for maximum tracking accuracy, enabled by GPU speed.
- **ONNX Runtime graph optimizations**: `ORT_ENABLE_ALL` graph optimization enabled for ArcFace session (constant folding, node fusion, layout optimization).
- **Git LFS fix in CI/CD**: Added `lfs: true` to `actions/checkout@v4` so model weights are included in Docker builds (previously only LFS pointer files were copied, causing `UnpicklingError` at startup).
- **Direct VM model injection**: Models can be hot-injected into a running container (`docker cp`) without rebuilding the image.

---

## 🐛 Deployment Challenges & How We Solved Them

### ❌ Docker container pulled but app crashed silently
**Cause:** Deployment script lacked `set -e` — `docker pull` failed silently and the script continued.  
**Fix:** Added `set -e` so any failing command immediately aborts the pipeline.

### ❌ Server crashed at startup when camera was offline
**Cause:** `stream.py` raised `RuntimeError` in `__init__` if OpenCV couldn't open RTSP.  
**Fix:** Changed hard raise to `logger.warning`. Server starts even if camera is temporarily offline.

### ❌ MJPEG stream showed black screen
**Cause:** No frames ever placed into the queue when camera was offline — browser waited forever.  
**Fix:** Added a generated "Camera Connecting or Offline" placeholder frame yielded when queue is empty.

### ❌ Cloudflare Tunnels blocked RTSP
**Cause:** Cloudflare Tunnels only proxy HTTP/HTTPS — raw TCP RTSP is blocked.  
**Fix:** Switched to Tailscale (WireGuard Layer 3 VPN) — carries raw TCP with no protocol restrictions.

### ❌ RTSP connected but no frames (UDP packet loss over VPN)
**Cause:** OpenCV defaulted to UDP RTSP transport. UDP packets drop over VPN due to MTU fragmentation.  
**Fix:** Set `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` to force reliable TCP transport.

### ❌ VM couldn't pull Docker image (IAM 403)
**Cause:** Compute Engine service account was missing `Artifact Registry Reader` role.  
**Fix:** Granted `artifactregistry.repositories.downloadArtifacts` permission to the service account.

### ❌ Models loaded as LFS pointer files (UnpicklingError at startup)
**Cause:** `actions/checkout@v4` without `lfs: true` downloads 132-byte pointer text files instead of real model weights. PyTorch tries to unpickle the pointer file and crashes.  
**Fix:** Added `lfs: true` to the checkout action in `.github/workflows/deploy.yml`.

### ❌ GitHub Action failed on VM SSH (wrong zone secret)
**Cause:** `GCE_VM_ZONE` GitHub secret was still set to the old US zone (`us-central1-a`) after migrating the VM to Tel Aviv (`me-west1-b`).  
**Fix:** Update the `GCE_VM_ZONE` secret to `me-west1-b` in GitHub Repository Settings → Secrets.

### ❌ Docker authentication failed during manual deployment
**Cause:** VM's Docker daemon wasn't authenticated to the Artifact Registry.  
**Fix:** Run `gcloud auth configure-docker me-west1-docker.pkg.dev` on the VM before `docker pull`.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Stream shows "Camera Connecting or Offline" | Check Tailscale is running (`tailscale status`); verify RTSP URL and camera IP |
| `AI FPS: 0.0` on stream | Model file is corrupt — check container logs for `UnpicklingError`; re-run `docker cp` with real model files |
| `/status` returns 500 | Check container logs for Python errors |
| `Frame queue timed out` in logs | RTSP dropping packets — ensure `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` is set |
| `Cannot connect to MongoDB` | Verify `MONGO_URI`; check Atlas Network Access allows the VM IP |
| Models load slowly on first start | First run loads ~175 MB into GPU memory — normal |
| Very low FPS (< 5) | Lower `FRAME_WIDTH`/`FRAME_HEIGHT`; verify GPU is accessible (`--gpus all` in docker run) |
| Tailscale VPN disconnects after reboot | Re-run `sudo tailscale up --accept-routes` on the VM |
| `git lfs pull` fails | Check GitHub LFS bandwidth quota — if exceeded, copy models manually via `gcloud compute scp` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
