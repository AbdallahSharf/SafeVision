"""
SafeVision entry point.

Starts the FastAPI/Uvicorn server which in turn launches the background
RTSP processing loop (see ``app.api``).
"""

import uvicorn

from app.config import settings, logger


def main() -> None:
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
