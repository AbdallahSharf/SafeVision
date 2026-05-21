import os
import logging
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
from insightface.model_zoo import get_model
from ultralytics import YOLO

from db import faces_collection

# ==========================
# LOGGING SETUP
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("safevision.log"),
    ],
)
logger = logging.getLogger(__name__)

# ==========================
# CONFIG (from env vars with safe defaults)
# ==========================
RTSP_URL        = os.environ.get("RTSP_URL", "rtsp://admin:admin1234@192.168.1.9:554/cam/realmonitor?channel=1&subtype=0")
DISPLAY_OUTPUT  = os.environ.get("DISPLAY_OUTPUT", "true").lower() == "true"  # set to "false" on headless server

YOLO_MODEL_PATH    = os.environ.get("YOLO_MODEL_PATH",    "models/best.pt")
ARCFACE_MODEL_PATH = os.environ.get("ARCFACE_MODEL_PATH", "models/w600k_r50.onnx")

YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.4"))
BOX_CONF_THRESHOLD  = float(os.environ.get("BOX_CONF_THRESHOLD",  "0.6"))
RECOG_THRESHOLD     = float(os.environ.get("RECOG_THRESHOLD",     "0.6"))

FRAME_WIDTH    = int(os.environ.get("FRAME_WIDTH",    "800"))
FRAME_HEIGHT   = int(os.environ.get("FRAME_HEIGHT",   "600"))
FACE_SIZE      = int(os.environ.get("FACE_SIZE",      "112"))
FACE_MARGIN    = int(os.environ.get("FACE_MARGIN",    "20"))
IMGSZ          = int(os.environ.get("IMGSZ",          "640"))
QUEUE_SIZE     = int(os.environ.get("QUEUE_SIZE",     "5"))
HISTORY_LEN    = int(os.environ.get("HISTORY_LEN",    "5"))
QUEUE_TIMEOUT  = float(os.environ.get("QUEUE_TIMEOUT", "2.0"))
DB_NUM_CANDIDATES = int(os.environ.get("DB_NUM_CANDIDATES", "100"))
DB_TOP_K          = int(os.environ.get("DB_TOP_K",          "5"))

# Low-light enhancement
LOW_LIGHT_ENABLE         = os.environ.get("LOW_LIGHT_ENABLE",         "true").lower() == "true"
LOW_LIGHT_AUTO_THRESHOLD = int(os.environ.get("LOW_LIGHT_AUTO_THRESHOLD", "80"))   # mean brightness 0-255; enhancement triggers below this
CLAHE_CLIP_LIMIT         = float(os.environ.get("CLAHE_CLIP_LIMIT",    "3.0"))     # 2.0–4.0; higher = stronger contrast boost
CLAHE_TILE_SIZE          = int(os.environ.get("CLAHE_TILE_SIZE",       "8"))        # CLAHE grid tile size
DENOISE_STRENGTH         = int(os.environ.get("DENOISE_STRENGTH",      "7"))        # 0=off, 3–10=light–strong; higher costs FPS

# ==========================
# STREAM CLASS
# ==========================
class VideoStream:
    def __init__(self, rtsp_url: str, queue_size: int = QUEUE_SIZE):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")

        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        logger.info("VideoStream started.")

    def _update(self):
        while self.running:
            try:
                if not self.cap.isOpened():
                    logger.warning("Stream lost — reconnecting...")
                    self.cap.open(self.rtsp_url)
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()

                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass

                self.q.put(frame)

            except Exception as e:
                logger.error("Stream thread error: %s", e)
                time.sleep(1)

    def read(self, timeout: float = QUEUE_TIMEOUT):
        """Return the latest frame, or None on timeout."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            logger.warning("Frame queue timed out — stream may be stalled.")
            return None

    def stop(self):
        self.running = False
        self._thread.join(timeout=3)
        self.cap.release()
        logger.info("VideoStream stopped.")


# ==========================
# LOAD MODELS
# ==========================
logger.info("Loading YOLO model: %s", YOLO_MODEL_PATH)
model = YOLO(YOLO_MODEL_PATH)
model.fuse()

logger.info("Loading ArcFace model: %s", ARCFACE_MODEL_PATH)
arcface = get_model(ARCFACE_MODEL_PATH)
arcface.prepare(ctx_id=-1)

# ==========================
# LOW-LIGHT ENHANCEMENT SETUP
# ==========================
clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
)

def is_dark(frame: np.ndarray) -> bool:
    """Return True if frame mean brightness is below the configured threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < LOW_LIGHT_AUTO_THRESHOLD

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full-frame low-light enhancement applied BEFORE YOLO detection.
    Pipeline: optional denoise -> CLAHE on L channel (LAB) -> gamma lift.
    Only activates when LOW_LIGHT_ENABLE=true AND frame is detected as dark.
    """
    if not LOW_LIGHT_ENABLE or not is_dark(frame):
        return frame

    # Step 1 — fast denoise (skip if DENOISE_STRENGTH=0 to preserve FPS)
    if DENOISE_STRENGTH > 0:
        frame = cv2.fastNlMeansDenoisingColored(
            frame, None,
            h=DENOISE_STRENGTH,
            hColor=DENOISE_STRENGTH,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # Step 2 — CLAHE on L channel of LAB (boosts contrast without blowing out color)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Step 3 — gamma correction to lift shadows (gamma=0.6 -> brighter midtones)
    inv_gamma = 1.0 / 0.6
    lut = np.array(
        [min(255, int((i / 255.0) ** inv_gamma * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    frame = cv2.LUT(frame, lut)

    return frame

def enhance_face(face: np.ndarray) -> np.ndarray:
    """
    Face-crop level enhancement applied BEFORE ArcFace embedding.
    Runs on every face crop when LOW_LIGHT_ENABLE=true — crops can be
    locally dark even if the overall frame passes the brightness threshold.
    """
    if not LOW_LIGHT_ENABLE:
        return face

    # CLAHE on L channel only — keeps color channels intact for ArcFace
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    face = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return face


# ==========================
# DB SANITY CHECK
# ==========================
face_count = faces_collection.count_documents({})
logger.info("Total faces in DB: %d", face_count)
if face_count == 0:
    logger.warning("Face DB is empty — all detections will be 'Unauthorized'.")

# ==========================
# RECOGNITION FUNCTION
# ==========================
def recognize_face_mongo(embedding: np.ndarray, threshold: float = RECOG_THRESHOLD):
    """
    Search MongoDB for the closest face embedding.
    Returns (name, score) — name is 'Unauthorized' when below threshold.
    """
    try:
        cursor = faces_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": embedding.tolist(),
                    "path": "embedding",
                    "numCandidates": DB_NUM_CANDIDATES,
                    "limit": DB_TOP_K,
                }
            },
            {
                "$project": {
                    "name": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ])

        results = list(cursor)

        if not results:
            return "Unauthorized", 0.0

        # Group by name, keep highest score per person
        scores_by_name: dict[str, float] = {}
        for match in results:
            name  = match.get("name", "Unknown")
            score = match.get("score", 0.0)
            if score > scores_by_name.get(name, 0.0):
                scores_by_name[name] = score

        best_name  = max(scores_by_name, key=scores_by_name.__getitem__)
        best_score = scores_by_name[best_name]

        if best_score < threshold:
            return "Unauthorized", best_score

        return best_name, best_score

    except Exception as e:
        logger.error("MongoDB search error: %s", e)
        return "Error", 0.0


# ==========================
# MAIN LOOP
# ==========================
def main():
    stream = VideoStream(RTSP_URL)
    history: deque[str] = deque(maxlen=HISTORY_LEN)

    fps       = 0.0
    fps_time  = time.time()
    frame_count = 0

    logger.info("Starting main loop. DISPLAY_OUTPUT=%s", DISPLAY_OUTPUT)

    try:
        while True:
            frame = stream.read()

            if frame is None:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = enhance_frame(frame)  # low-light boost before detection

            results = model(frame, conf=YOLO_CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)

            for r in results:
                boxes = r.boxes

                for i in range(len(boxes)):
                    if float(boxes.conf[i]) < BOX_CONF_THRESHOLD:
                        continue

                    cls   = int(boxes.cls[i])
                    label = model.names.get(cls, "")

                    if label.lower() != "face":
                        continue

                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                    h, w = frame.shape[:2]
                    x1 = max(0, x1 - FACE_MARGIN)
                    y1 = max(0, y1 - FACE_MARGIN)
                    x2 = min(w, x2 + FACE_MARGIN)
                    y2 = min(h, y2 + FACE_MARGIN)

                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Preprocessing — enhance face crop for low-light conditions
                    face        = enhance_face(face)
                    face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
                    face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                    # Embedding
                    embedding = arcface.get_feat(face_rgb).flatten()
                    norm = np.linalg.norm(embedding)
                    if norm == 0:
                        continue
                    embedding = embedding / norm

                    # Recognition
                    identity, score = recognize_face_mongo(embedding)

                    # Temporal smoothing
                    history.append(identity)
                    identity = max(set(history), key=history.count)

                    color = (0, 255, 0) if identity != "Unauthorized" else (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{identity} ({score:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            # FPS counter
            frame_count += 1
            if frame_count >= 10:
                now        = time.time()
                fps        = frame_count / (now - fps_time)
                fps_time   = now
                frame_count = 0

            if DISPLAY_OUTPUT:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("SafeVision AI", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit signal received.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception("Fatal error in main loop: %s", e)
    finally:
        stream.stop()
        if DISPLAY_OUTPUT:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()