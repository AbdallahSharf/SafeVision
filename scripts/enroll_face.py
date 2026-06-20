"""
SafeVision — Face Enrollment Script
====================================
Enroll one or more people into the SafeVision face database.

For best recognition accuracy, provide 5–10 images of each person taken
from different angles and under different lighting conditions.  The script
processes every image, extracts an ArcFace embedding, and inserts it into
MongoDB.  The live system will then recognise that person automatically.

Usage
-----
# Enroll a single image:
    python scripts/enroll_face.py --name "Ahmad" --images path/to/ahmad.jpg

# Enroll all images in a folder (recommended — more angles = better accuracy):
    python scripts/enroll_face.py --name "Ahmad" --images path/to/ahmad/

# Preview what would be enrolled without writing to DB (dry run):
    python scripts/enroll_face.py --name "Ahmad" --images path/to/folder/ --dry-run

# List everyone currently enrolled:
    python scripts/enroll_face.py --list

# Remove all embeddings for a person:
    python scripts/enroll_face.py --delete "Ahmad"

Requirements
------------
Run from the project root so that the ``app`` package is importable:
    cd SafeVision
    python scripts/enroll_face.py ...

Make sure your .env file is present and MONGO_URI is set.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Ensure project root is on sys.path ────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.database import faces_collection
from app.models_loader import get_arcface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_image_paths(source: str) -> list[Path]:
    """Return a sorted list of image paths from a file or directory."""
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"⚠️  '{p.name}' is not a supported image type. Skipping.")
            return []
        return [p]
    if p.is_dir():
        paths = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            print(f"⚠️  No images found in '{source}'. Supported: {SUPPORTED_EXTENSIONS}")
        return paths
    print(f"❌  '{source}' is not a valid file or directory.")
    sys.exit(1)


def _embed(image_path: Path, arcface) -> np.ndarray | None:
    """
    Load an image, detect the largest face region using YOLO (if available), 
    and compute ArcFace embedding using the exact same resizing math as the live stream.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️  Could not read '{image_path.name}' — skipping.")
        return None

    h, w = img.shape[:2]
    
    # Check if the image is already a tight crop (like those from unauthorized_faces)
    # If it's a small image (e.g. < 400x400), it's likely a crop. We skip YOLO to avoid 
    # detecting a face inside a face, and just use the whole image like the live stream does.
    face_resized = None
    if w < 400 and h < 400:
        face_crop = img
    else:
        # For large images, try to use YOLO to match the live stream bounding boxes
        try:
            from app.models_loader import get_yolo
            from app.config import settings
            from app.processor import align_face
            
            yolo = get_yolo()
            # Run YOLO exactly like processor.py does
            yolo_results = yolo(img, verbose=False, conf=settings.YOLO_CONF_THRESHOLD)
            r = yolo_results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            
            if len(boxes) > 0:
                # Get the largest box
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = np.argmax(areas)
                
                # Check for keypoints and align
                kpts = None
                if hasattr(r, 'keypoints') and r.keypoints is not None:
                    if len(r.keypoints.xy) > largest_idx and r.keypoints.xy[largest_idx].shape[0] == 5:
                        kpts = r.keypoints.xy[largest_idx].cpu().numpy()
                
                if kpts is not None:
                    try:
                        face_resized = align_face(img, kpts)
                    except Exception as e:
                        print(f"  ⚠️  Face alignment failed ({e}) — falling back to crop.")
                
                if face_resized is None:
                    x1, y1, x2, y2 = boxes[largest_idx].astype(int)
                    # Apply the same margin as processor.py
                    x1 = max(0, x1 - settings.FACE_MARGIN)
                    y1 = max(0, y1 - settings.FACE_MARGIN)
                    x2 = min(w, x2 + settings.FACE_MARGIN)
                    y2 = min(h, y2 + settings.FACE_MARGIN)
                    face_crop = img[y1:y2, x1:x2]
            else:
                print(f"  ℹ️  No face detected by YOLO in '{image_path.name}' — using full image.")
                face_crop = img
        except Exception as e:
            print(f"  ⚠️  YOLO failed ({e}) — using full image.")
            face_crop = img

    if face_resized is None:
        if face_crop.size == 0:
            print(f"  ⚠️  Empty crop for '{image_path.name}' — skipping.")
            return None

        # Resize to exactly 112x112 with stretching, matching the live stream's cv2.resize
        face_resized = cv2.resize(face_crop, (112, 112))
    
    # Check sharpness — warn if blurry (Laplacian variance)
    blur_score = cv2.Laplacian(face_resized, cv2.CV_64F).var()
    if blur_score < 40:
        print(f"  ⚠️  '{image_path.name}' is very blurry (score={blur_score:.0f}) — embedding may be unreliable.")

    # Convert to RGB and embed
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Apply the same enhancement as live stream!
    from app.enhancement import enhance_face
    face_enhanced = enhance_face(face_rgb)

    embedding = arcface.get_feat(face_enhanced).flatten()
    norm = np.linalg.norm(embedding)
    if norm == 0:
        print(f"  ⚠️  Zero-norm embedding for '{image_path.name}' — skipping.")
        return None

    return embedding / norm


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_enroll(name: str, source: str, dry_run: bool) -> None:
    """Enroll all images from *source* under the name *name*."""
    print(f"\n🔍  Loading images from: {source}")
    image_paths = _collect_image_paths(source)
    if not image_paths:
        sys.exit(1)

    print(f"📦  Found {len(image_paths)} image(s). Loading ArcFace model …")
    arcface = get_arcface()

    inserted = 0
    failed = 0
    for img_path in image_paths:
        print(f"\n  Processing: {img_path.name}")
        embedding = _embed(img_path, arcface)
        if embedding is None:
            failed += 1
            continue

        doc = {
            "name": name,
            "embedding": embedding.tolist(),
            "source_file": img_path.name,
            "enrolled_at": time.time(),
        }

        if dry_run:
            print(f"  ✅ [DRY RUN] Would insert embedding (norm={np.linalg.norm(embedding):.4f})")
        else:
            faces_collection.insert_one(doc)
            print(f"  ✅ Enrolled embedding (norm={np.linalg.norm(embedding):.4f})")
        inserted += 1

    # Summary
    print(f"\n{'─' * 50}")
    if dry_run:
        print(f"DRY RUN complete — {inserted} embedding(s) would be inserted for '{name}'.")
    else:
        print(f"✅  Done — inserted {inserted} embedding(s) for '{name}'.")
        if failed:
            print(f"⚠️   {failed} image(s) could not be processed.")
        total = faces_collection.count_documents({"name": name})
        print(f"📊  Total embeddings in DB for '{name}': {total}")


def cmd_list() -> None:
    """List all enrolled persons and their embedding counts."""
    pipeline = [
        {"$group": {"_id": "$name", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    results = list(faces_collection.aggregate(pipeline))
    if not results:
        print("\n📭  No faces enrolled yet.")
        return

    print(f"\n{'─' * 40}")
    print(f"  {'Name':<25} {'Embeddings':>10}")
    print(f"{'─' * 40}")
    total = 0
    for r in results:
        print(f"  {r['_id']:<25} {r['count']:>10}")
        total += r["count"]
    print(f"{'─' * 40}")
    print(f"  {'TOTAL':<25} {total:>10}")
    print(f"{'─' * 40}\n")


def cmd_delete(name: str) -> None:
    """Remove all embeddings for *name* from the database."""
    count = faces_collection.count_documents({"name": name})
    if count == 0:
        print(f"\n⚠️  No embeddings found for '{name}'.")
        return

    confirm = input(f"\n⚠️  This will delete {count} embedding(s) for '{name}'. Type the name to confirm: ")
    if confirm.strip() != name:
        print("Cancelled.")
        return

    result = faces_collection.delete_many({"name": name})
    print(f"🗑️   Deleted {result.deleted_count} embedding(s) for '{name}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SafeVision face enrollment tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command")

    # enroll
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a person's face(s)")
    enroll_parser.add_argument("--name", required=True, help="Person's display name")
    enroll_parser.add_argument("--images", required=True, help="Path to image file or folder")
    enroll_parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")

    # list
    subparsers.add_parser("list", help="List all enrolled persons")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Remove all embeddings for a person")
    delete_parser.add_argument("name", help="Person's name to delete")

    # Legacy flat-flag support (--name, --list, --delete) for backward compat
    parser.add_argument("--name", help="Person's display name (legacy)")
    parser.add_argument("--images", help="Path to image file or folder (legacy)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing (legacy)")
    parser.add_argument("--list", action="store_true", help="List enrolled persons (legacy)")
    parser.add_argument("--delete", metavar="NAME", help="Delete a person (legacy)")

    args = parser.parse_args()

    # Route to command
    if args.command == "enroll":
        cmd_enroll(args.name, args.images, args.dry_run)
    elif args.command == "list":
        cmd_list()
    elif args.command == "delete":
        cmd_delete(args.name)
    # Legacy flat flags
    elif getattr(args, "list", False):
        cmd_list()
    elif getattr(args, "delete", None):
        cmd_delete(args.delete)
    elif getattr(args, "name", None) and getattr(args, "images", None):
        cmd_enroll(args.name, args.images, args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
