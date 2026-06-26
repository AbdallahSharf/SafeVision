"""
SafeVision entry point.

Starts the FastAPI/Uvicorn server which in turn launches the background
RTSP processing loop (see ``app.api``).
"""

import os
import sys
import subprocess
import uvicorn

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings, logger


def main() -> None:
    engine_path = "models/best.engine"
    if not os.path.exists(engine_path):
        logger.info("YOLO TensorRT engine not found. Building it now (this may take a few minutes)...")
        try:
            subprocess.run([sys.executable, "scripts/export_tensorrt.py"], check=True)
        except Exception as e:
            logger.error(f"Failed to build TensorRT engine: {e}")
            logger.info("Continuing with PyTorch/ONNX backend instead.")
            
    logger.info(
        "Starting SafeVision API server on port %d …",
        settings.PORT,
    )
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
