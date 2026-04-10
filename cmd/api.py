from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
import asyncio
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to PYTHONPATH to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from internal.api.routes import create_router
from internal.core.camera_manager import CameraManager
from internal.core.predictor_wrapper import PredictorWrapper
from internal.utils.config_loader import load_config
from internal.utils.logging_utils import setup_logging

logger = logging.getLogger("API")


def create_app():
    setup_logging()

    # Load configuration
    config = load_config("configs/config.yml")

    predictor_wrapper = PredictorWrapper(
        cfg_path=config.system.config_path,
        device=config.system.device,
    )
    manager = CameraManager(predictor_wrapper, config)

    # Lifespan handler
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            # Create a task for manager.start to avoid blocking startup
            # if it takes too long to initialize cameras or models
            start_task = asyncio.create_task(asyncio.to_thread(manager.start))

            yield
        except Exception as e:
            logger.exception(f"Unknown error in lifespan: {e}")
        finally:
            # Shutdown
            logger.info("Stopping camera manager...")
            await asyncio.to_thread(manager.stop)
            # Cancel start task if it's still running
            if not start_task.done():
                start_task.cancel()

    app = FastAPI(title="Paddle Detection API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_router = create_router(config)
    app.include_router(api_router, prefix="/api")

    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok", "message": "Paddle Detection API is running"}

    return app


if __name__ == "__main__":
    try:
        app = create_app()

        # Use env vars for config
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        # Note: Workers should be 1 because FaceRecognition is stateful (holds camera connections)
        uvicorn.run(
            app, host=host, port=port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info")
        )
    except Exception as e:
        logger.exception(f"Failed to start API: {e}")
        sys.exit(1)
