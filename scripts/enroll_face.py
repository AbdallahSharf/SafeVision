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
    Load an image, detect the largest face region, compute ArcFace embedding.

    Returns L2-normalised 512-d embedding, or None if no face can be processed.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️  Could not read '{image_path.name}' — skipping.")
        return None

    # Use a simple face crop heuristic: resize to 112×112 centered
    # (works well when the image is a portrait / passport-style photo)
    # If you have RetinaFace available, replace this with proper detection.
    h, w = img.shape[:2]

    # Try to detect a face using OpenCV's built-in Haar cascade as a fallback
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        # Use the largest detected face
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces_sorted[0]
        # Add 20% margin
        margin_x = int(fw * 0.20)
        margin_y = int(fh * 0.20)
        fx = max(0, fx - margin_x)
        fy = max(0, fy - margin_y)
        fw = min(w - fx, fw + 2 * margin_x)
        fh = min(h - fy, fh + 2 * margin_y)
        face_crop = img[fy:fy + fh, fx:fx + fw]
    else:
        # No face detected — use center crop (works for tightly-cropped portrait photos)
        print(f"  ℹ️  No face detected in '{image_path.name}' — using center crop.")
        short = min(h, w)
        cy, cx = h // 2, w // 2
        face_crop = img[cy - short // 2:cy + short // 2, cx - short // 2:cx + short // 2]

    if face_crop.size == 0:
        print(f"  ⚠️  Empty crop for '{image_path.name}' — skipping.")
        return None

    # Check sharpness — warn if blurry (Laplacian variance)
    blur_score = cv2.Laplacian(face_crop, cv2.CV_64F).var()
    if blur_score < 40:
        print(f"  ⚠️  '{image_path.name}' is very blurry (score={blur_score:.0f}) — embedding may be unreliable.")

    # Resize to ArcFace input size and convert to RGB
    face_resized = cv2.resize(face_crop, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    embedding = arcface.get_feat(face_rgb).flatten()
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
