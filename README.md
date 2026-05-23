# 🛡️ SafeVision

**Real-time AI-powered face recognition security system.**

SafeVision processes a live RTSP camera feed, detects faces using a custom YOLO model, recognizes identities via ArcFace embeddings matched against a MongoDB Atlas vector database, and streams the annotated video to mobile applications in real time.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![GCP](https://img.shields.io/badge/Google%20Cloud-GCE-4285F4?logo=googlecloud)
![Tailscale](https://img.shields.io/badge/Tailscale-VPN-0e8a8a?logo=tailscale)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

```
┌──────────────┐  RTSP (TCP)  ┌─────────────────────────────────────┐
│  IP Camera   │─────────────▶│         SafeVision Server           │
│ 192.168.1.9  │  Tailscale   │                                     │
└──────────────┘    VPN       │  ┌───────────┐   ┌──────────────┐  │
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
- **Offline camera resilience** — Server stays online and shows a placeholder if the camera disconnects
- **REST API** — Health checks, status metrics, and recent faces endpoint
- **Docker-ready** — Single command to build and deploy
- **CI/CD** — GitHub Actions auto-deploys to Google Compute Engine on every push to `main`
- **Secure camera tunnel** — Tailscale VPN connects the cloud server to the local camera without port forwarding

---

## 🧰 Tech Stack & Why We Chose It

### 🐍 Python 3.10
The de facto language for AI/ML. Its ecosystem — PyTorch, OpenCV, InsightFace — made it the only practical choice. Version 3.10 specifically was chosen for its union type hints (`X | Y`) and stability with the pinned library versions.

### ⚡ FastAPI
FastAPI serves dual duty: it streams MJPEG video frames and exposes REST endpoints. We chose it over Flask because:
- **Async by default** — Non-blocking request handling is critical for a streaming server.
- **Automatic OpenAPI docs** — `/docs` is available out-of-the-box for free.
- **Pydantic validation** — Strict typing catches bugs early.

### 🤖 YOLOv11 (Ultralytics) — Face Detection
YOLO (You Only Look Once) is a single-shot object detector that runs in real time even on CPU. We use a **custom-trained** YOLO model (`best.pt`) specifically fine-tuned on face data, so it detects faces with much higher precision than a generic YOLO model. It outperforms alternatives like Haar cascades (too many false positives) and MTCNN (too slow for real-time video).

### 🧠 ArcFace via InsightFace (ONNX) — Face Recognition
ArcFace is a state-of-the-art face recognition model that produces 512-dimensional embeddings. We load the pre-trained `w600k_r50.onnx` (trained on 600k identities, ResNet-50 backbone) via ONNX Runtime. The reason for ONNX is portability: the same weights run identically on CPU or GPU without code changes, and ONNX Runtime is significantly faster than a full PyTorch inference path for single-image inference.

### 🍃 MongoDB Atlas — Vector Database
After ArcFace produces a 512-float embedding for a detected face, we need to find the closest match in our identity database. MongoDB Atlas's **$vectorSearch** operator does this in a single database query using Approximate Nearest Neighbour (ANN) search — no separate vector database like Pinecone or Weaviate is needed. This simplifies the stack and reduces cost.

### 🐳 Docker
The entire application (Python, OpenCV, PyTorch, models) is containerized into a single image. This guarantees identical behavior between local development and the cloud VM, and eliminates the classic "works on my machine" problem. The image is stored in **Google Artifact Registry**.

### ☁️ Google Compute Engine (GCE)
The cloud VM that runs the SafeVision Docker container. A GCE VM was chosen over serverless options (Cloud Run, Lambda) because:
- RTSP streaming requires a **persistent, long-lived process** — serverless functions time out.
- The VM has a **stable public IP** that the mobile app can always connect to.

### 🔄 GitHub Actions — CI/CD
On every push to `main`, the pipeline automatically:
1. Builds a new Docker image.
2. Pushes it to Google Artifact Registry.
3. SSHs into the GCE VM and hot-swaps the container with zero downtime.

Authentication uses **Workload Identity Federation** (keyless auth) — no long-lived JSON service account keys are stored as secrets.

### 🔒 Tailscale — Secure Camera VPN
The IP camera sits on a private home network (`192.168.1.9`). The GCE VM is on Google Cloud. To connect them securely without opening ports on the router, we use **Tailscale** — a zero-config VPN built on WireGuard. The local Windows PC acts as a **subnet router**, advertising `192.168.1.0/24` to the Tailscale network so the GCE VM can reach the camera directly over an encrypted tunnel.

---

## 📁 Project Structure

```
SafeVision/
├── app/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Centralized config (env vars + dotenv)
│   ├── database.py          # MongoDB Atlas connection
│   ├── models_loader.py     # Lazy YOLO + ArcFace loading
│   ├── enhancement.py       # Low-light image enhancement (CLAHE)
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
- **RTSP camera** accessible via network (local or Tailscale VPN)
- **Git LFS** installed (for model files)

### 1. Clone the repository

```bash
git clone https://github.com/AbdallahSharf/SafeVision.git
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

### 2. Allow HTTP traffic on port 8080

```bash
gcloud compute firewall-rules create allow-safevision \
    --allow=tcp:8080 \
    --target-tags=http-server \
    --description="Allow SafeVision API access"
```

### 3. Create the `.env` file on the VM

```bash
# SSH into the VM
gcloud compute ssh safevision-vm --zone=me-west1-a

# Create env file directory
sudo mkdir -p /opt/safevision
sudo nano /opt/safevision/.env
# Paste your MONGO_URI and RTSP_URL, then save
```

### 4. Set up Tailscale VPN (for local camera access)

See the [Tailscale Setup](#-tailscale-vpn-setup) section below.

### 5. Deploy via GitHub Actions

Push to `main` — the CI/CD pipeline handles everything else automatically.

### 6. Access the stream

```
http://<VM-EXTERNAL-IP>:8080/stream
```

---

## 🔄 CI/CD (GitHub Actions)

Auto-deploys to GCE on every push to `main` using **Workload Identity Federation** (no JSON keys required).

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID (e.g. `safevision-00`) |
| `GCP_WIF_PROVIDER` | Workload Identity Provider resource name |
| `GCP_SA_EMAIL` | Service account email |
| `GCE_VM_NAME` | Compute Engine VM name (e.g. `safevision-vm`) |
| `GCE_VM_ZONE` | VM zone (e.g. `me-west1-a`) |
| `GCE_SSH_PRIVATE_KEY` | Private SSH key for connecting to the VM |

> **Note:** The `.env` file (containing `MONGO_URI` and `RTSP_URL`) lives directly on the VM at `/opt/safevision/.env` and is **not** stored in GitHub secrets. The deployment pipeline mounts it into the container at runtime.

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

### 3. Install Tailscale on the VM and link it

```bash
# Install
ssh user@<VM-IP> "curl -fsSL https://tailscale.com/install.sh | sh"

# Start and get the auth link
ssh user@<VM-IP> "sudo tailscale up"
# → Opens a URL — open it and log in with the SAME account as Step 1

# Tell the VM to use routes from the PC
ssh user@<VM-IP> "sudo tailscale up --accept-routes"
```

### 4. Update RTSP URL

Once connected, update your VM's `/opt/safevision/.env` to use the camera's local IP:

```env
RTSP_URL=rtsp://admin:your_password@192.168.1.9:554/cam/realmonitor?channel=1&subtype=0
```

Restart the container:
```bash
docker restart safevision
```

---

## 🐛 Deployment Challenges & How We Solved Them

This section documents the real problems encountered during deployment and the solutions applied — useful for anyone setting up a similar system.

### ❌ Problem 1: Docker container pulled but the app crashed silently
**What happened:** The GitHub Actions workflow reported ✅ success even though the container wasn't actually running. The deployment script lacked `set -e`, so even when `docker pull` failed silently, the script continued to the final "success" echo.

**Fix:** Added `set -e` to the SSH deployment script so any failing command immediately aborts the pipeline with a red failure status.

---

### ❌ Problem 2: Server crashed at startup when camera was offline
**What happened:** `stream.py` raised a `RuntimeError` in `__init__` if OpenCV couldn't open the RTSP stream. This caused FastAPI's startup lifecycle to fail, taking down the entire server.

**Fix:** Changed the hard `raise RuntimeError(...)` to a `logger.warning(...)`. The server now starts successfully even if the camera is temporarily offline, and the background thread keeps retrying the connection automatically.

---

### ❌ Problem 3: `/stream` endpoint showed a black screen (infinite loading)
**What happened:** Because the camera was offline, no frames were ever placed into the MJPEG queue. The browser waited forever for the first multipart boundary to arrive, displaying a frozen blank page.

**Fix:** Added a generated placeholder frame ("Camera Connecting or Offline" on a black background) that is yielded by the MJPEG generator whenever the frame queue is empty, so the browser always gets an immediate response.

---

### ❌ Problem 4: Cloudflare Tunnels don't support RTSP
**What happened:** The initial approach used a Cloudflare Quick Tunnel (`trycloudflare.com`) to expose the camera. Cloudflare Tunnels only proxy HTTP/HTTPS traffic — raw TCP streams like RTSP are blocked at the protocol level, causing a 30-second timeout in OpenCV every attempt.

**Fix:** Switched to **Tailscale** as the camera tunnel. Tailscale uses WireGuard (Layer 3 VPN), which carries raw TCP at the network level and has no restrictions on protocols. The local Windows PC acts as a subnet router, and the GCE VM connects to the camera's local IP (`192.168.1.9`) through the encrypted tunnel.

---

### ❌ Problem 5: RTSP stream connected but no frames arrived (UDP packet loss)
**What happened:** Even after Tailscale connected the VM to the camera's local IP, OpenCV defaulted to UDP transport for RTSP. UDP packets are frequently dropped over VPN tunnels due to MTU fragmentation and encapsulation overhead, causing the stream thread to time out on every read attempt.

**Fix:** Added `os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"` in `stream.py` before importing `cv2`. This forces FFMPEG to use TCP as the RTSP transport layer, which is reliable over VPN tunnels and eliminates packet loss.

---

### ❌ Problem 6: VM couldn't pull the Docker image (IAM permission error)
**What happened:** The GCE VM's default Compute Engine service account was missing the `Artifact Registry Reader` role, so `docker pull` failed with a 403 Unauthorized error. The original deployment script used `gcloud auth configure-docker` inside the VM, but the VM's service account lacked the necessary scope.

**Fix:** Granted the `artifactregistry.repositories.downloadArtifacts` permission to the Compute Engine service account via Google Cloud IAM. Then restructured the deployment to pass an access token from the GitHub Actions runner (which already has the correct credentials) to the VM via the SSH session for the `docker login` step.

---

## ⚙️ Environment Variables

See [`.env.example`](.env.example) for all available variables with descriptions.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGO_URI` | ✅ | — | MongoDB Atlas connection string |
| `RTSP_URL` | ✅ | — | RTSP camera URL |
| `PORT` | ❌ | `8080` | API server port |
| `DISPLAY_OUTPUT` | ❌ | `false` | Enable OpenCV GUI window |
| `YOLO_CONF_THRESHOLD` | ❌ | `0.4` | YOLO detection confidence threshold |
| `BOX_CONF_THRESHOLD` | ❌ | `0.6` | Bounding box confidence filter |
| `RECOG_THRESHOLD` | ❌ | `0.6` | Face recognition match threshold |
| `LOW_LIGHT_ENABLE` | ❌ | `true` | Auto low-light enhancement (CLAHE) |
| `HISTORY_LEN` | ❌ | `5` | Frames used for temporal identity smoothing |
| `FRAME_WIDTH` | ❌ | `800` | Processing resolution width |
| `FRAME_HEIGHT` | ❌ | `600` | Processing resolution height |

---

## 🚀 Recent Improvements (v2.0)

We recently shipped a major update to the SafeVision pipeline to dramatically improve performance, accuracy, and reliability:

### ⚡ Performance Architecture
- **Async 3-Stage Pipeline**: Decoupled the RTSP reader, YOLO detector, and ArcFace recognizer into separate concurrent threads connected by bounded queues. This pipelining roughly doubles throughput on multi-core CPUs.
- **IoU-based Object Tracking (SORT-lite)**: Replaced grid-based spatial smoothing with a proper multi-object tracker. Each face gets a persistent integer ID across frames, eliminating "identity flicker" and correctly handling people crossing paths.
- **Frame Skipping for Detection & Recognition**: YOLO detection and ArcFace recognition only run every N frames (e.g., every 3rd frame), reusing cached bounding boxes and identities in between. This drastically cuts CPU/GPU and MongoDB Vector Search load while keeping the stream perfectly smooth, eliminating frame drops.
- **JPEG Quality Tuning**: MJPEG stream quality tuned from 80 to 65, reducing per-frame bandwidth by ~35% for a much smoother stream over the Tailscale VPN.

### 🎯 Accuracy & Reliability
- **Multi-angle Enrollment CLI**: Added `scripts/enroll_face.py`, allowing bulk enrollment of multiple photos per identity to improve recognition accuracy for non-frontal faces.
- **Adaptive Recognition Thresholds**: Instead of a fixed global threshold, the system now computes a per-identity calibrated threshold based on the variance of their enrolled embeddings (tighter gates for consistent faces, looser gates for varied faces).
- **Blur Quality Gate**: Computes a Laplacian variance score on face crops before recognition. Blurry/motion-blurred faces are skipped, preventing low-quality embeddings from polluting the temporal smoothing history.
- **Persistent Tailscale**: Added a pre-auth key to the GitHub Actions deployment script so the VPN reconnects automatically after a VM reboot.

### 🔐 Security & Infrastructure
- **API Authentication**: Added HTTP Bearer Token checks (via FastAPI's `Depends`) to secure the `/stream`, `/faces`, and `/admin` endpoints.
- **Real-time Push Alerts**: Added Firebase Cloud Messaging (FCM) integration. A webhook fires the exact millisecond an "Unauthorized" face is detected, sending a push notification and confidence score directly to the mobile app.
- **GPU Deployment**: Migrated the deployment to a Google Cloud `europe-west4-a` Spot VM equipped with an NVIDIA Tesla T4 GPU. Updated the `Dockerfile` and CI/CD script to utilize the GPU via `--gpus all`.

---

## 🔮 Future Roadmap & Cost-Saving Tips

### 💰 Cost-Saving Tips for Google Cloud
If you are running this as a personal or home project, you can drastically reduce your Google Cloud bill:
1. **Use Spot Instances (-70% cost):** Change your VM Provisioning Model to "Spot". Google will run your VM on excess capacity at a huge discount. Because SafeVision uses Docker with `--restart unless-stopped`, if Google restarts your VM, the camera stream will automatically come back online.
2. **Release your Static IP:** You are paying ~$3/mo for a static IP. Because the VM is connected to Tailscale, you can connect to the Tailscale `100.x.x.x` IP for free and release the public static IP.
3. **Downgrade to Standard HDD:** You are currently using a 50GB SSD (~$8.50/mo). Since the AI models load directly into RAM on startup, you can recreate the VM using a Standard Persistent Disk to drop this cost to ~$2.00/mo without losing stream performance.

### 🎯 Further Recommendations for Performance & Accuracy
**Performance:**
1. **Asynchronous Database Queries** — Move the MongoDB `$vectorSearch` into an asynchronous background task so it never blocks the main video rendering thread, completely decoupling recognition from video FPS.
2. **TensorRT Optimization** — Compile the YOLO and ArcFace ONNX models into NVIDIA TensorRT engines. This can double the inference speed on the T4 GPU compared to standard PyTorch.
3. **WebRTC instead of MJPEG** — Migrate the stream endpoint to WebRTC (`aiortc`) for adaptive bitrate, sub-200ms latency, and better browser support over Tailscale.

**Accuracy:**
1. **Fine-Tune ArcFace** — Fine-tune the ArcFace model on low-angle, low-light security camera footage (currently it is trained on high-quality frontal faces like WebFace600k).
2. **Specialized Dense Face Detector** — Replace the standard YOLO detector with RetinaFace or YOLOv11-Face to detect 5 facial landmarks (eyes, nose, mouth corners), enabling precise geometric alignment before ArcFace embedding.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Stream shows "Camera Connecting or Offline" | Check Tailscale is running on your PC (`tailscale status`); verify the camera local IP is correct |
| `Frame queue timed out` in logs | The RTSP stream is dropping packets — ensure TCP transport is forced (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` in `.env`) |
| `Cannot connect to MongoDB` | Verify `MONGO_URI`; check Atlas Network Access allows the VM's IP or `0.0.0.0/0` for development |
| Models load slowly on first start | First run loads ~170 MB of weights into RAM — subsequent starts are faster due to OS file caching |
| Very low FPS (< 5) | Lower `FRAME_WIDTH`/`FRAME_HEIGHT`; set `DENOISE_STRENGTH=0`; consider upgrading to a GPU VM |
| `cv2.imshow` crash on server | Ensure `DISPLAY_OUTPUT=false` (default in Docker) |
| Tailscale VPN disconnects after reboot | Re-run `sudo tailscale up --accept-routes` on the VM, or set up a systemd service with a pre-auth key |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
