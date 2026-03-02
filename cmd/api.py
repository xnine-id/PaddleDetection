from contextlib import asynccontextmanager
import logging
import os
import sys
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to PYTHONPATH to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from internal.api.routes import create_router
from internal.core.camera_manager import CameraManager
from internal.core.fight_detector import FightDetector
from internal.utils.config_loader import load_config
from internal.utils.logging_utils import setup_logging

logger = logging.getLogger("API")

def create_app():
    setup_logging()

    # Load configuration
    config = load_config('configs/config.yml')

    fight_detector = FightDetector(
        cfg_path=config['paddle_detection']['config_path'],
        device=config['paddle_detection']["device"],
    )
    manager = CameraManager(fight_detector, config)

    # Lifespan handler
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            # Startup
            manager.start()
            
            yield
            
            # Shutdown
            manager.stop()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Unknown error in lifespan: {e}")
        finally:
            manager.stop()

    app = FastAPI(title="Fighting Detection API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_router = create_router(fight_detector, config)
    app.include_router(api_router, prefix="/api")

    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok", "message": "Fighting Detection API is running"}

    return app

if __name__ == "__main__":
    try:
        app = create_app()
        
        # Use env vars for config
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        
        # Note: Workers should be 1 because FaceRecognition is stateful (holds camera connections)
        uvicorn.run(app, host=host, port=port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        sys.exit(1)
