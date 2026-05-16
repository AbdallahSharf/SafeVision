from ultralytics import YOLO
import cv2
import numpy as np
from insightface.model_zoo import get_model
import threading
import queue
import time
from db import faces_collection
from collections import deque

# smoothing buffer
history = deque(maxlen=5)

# ==========================
# STREAM CLASS
# ==========================
class VideoStream:
    def __init__(self, rtsp_url, queue_size=5):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open RTSP stream")

        self.q = queue.Queue(maxsize=queue_size)
        self.running = True

        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            try:
                if not self.cap.isOpened():
                    print("⚠️ Reconnecting...")
                    self.cap.open(self.rtsp_url)
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()

                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                if self.q.full():
                    self.q.get()

                self.q.put(frame)

            except Exception as e:
                print("❌ Thread error:", e)
                time.sleep(1)

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.cap.release()


# ==========================
# LOAD MODELS
# ==========================
model = YOLO("best.pt")
model.fuse()

arcface = get_model("w600k_r50.onnx")
arcface.prepare(ctx_id=-1)

# ==========================
# DEBUG: CHECK DB
# ==========================
print("Total faces in DB:", faces_collection.count_documents({}))

# ==========================
# RTSP STREAM
# ==========================
rtsp_url = "rtsp://admin:admin1234@192.168.1.6:554/cam/realmonitor?channel=1&subtype=0"
stream = VideoStream(rtsp_url)

# ==========================
# RECOGNITION FUNCTION
# ==========================
def recognize_face_mongo(embedding, threshold=0.6):
    try:
        results = faces_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": embedding.tolist(),
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "name": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ])

        result = list(results)

        print("Top matches:", result)  # 🔥 DEBUG

        if len(result) == 0:
            return "Unauthorized", 0

        scores_by_name = {}

        for match in result:
            name = match.get("name", "Unknown")
            score = match.get("score", 0)

            if name not in scores_by_name:
                scores_by_name[name] = []

            scores_by_name[name].append(score)

        best_name = "Unauthorized"
        best_score = 0

        for name, scores in scores_by_name.items():
            avg_score = max(scores)  # ✅ SIMPLIFIED

            if avg_score > best_score:
                best_score = avg_score
                best_name = name

        if best_score < threshold:
            return "Unauthorized", best_score

        return best_name, best_score

    except Exception as e:
        print("❌ Mongo search error:", e)
        return "Error", 0


# ==========================
# MAIN LOOP
# ==========================
fps_time = time.time()
frame_count = 0

while True:
    frame = stream.read()

    if frame is None:
        continue

    frame = cv2.resize(frame, (800, 600))

    results = model(frame, conf=0.4, imgsz=640, verbose=False)

    for r in results:
        boxes = r.boxes

        for i in range(len(boxes)):
            if boxes.conf[i] < 0.6:
                continue

            cls = int(boxes.cls[i])
            label = model.names[cls]

            if label.lower() != "face":
                continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            h, w, _ = frame.shape
            margin = 20

            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # ==========================
            # FIXED PREPROCESSING
            # ==========================
            face = cv2.resize(face, (112, 112))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # ==========================
            # EMBEDDING
            # ==========================
            embedding = arcface.get_feat(face).flatten()
            embedding = embedding / np.linalg.norm(embedding)

            # ==========================
            # RECOGNITION
            # ==========================
            identity, score = recognize_face_mongo(embedding)

            # smoothing
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
                2
            )

    # FPS
    frame_count += 1
    if frame_count >= 10:
        now = time.time()
        fps = frame_count / (now - fps_time)
        fps_time = now
        frame_count = 0
    else:
        fps = 0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("SafeVision AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()