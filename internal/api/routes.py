from internal.core.predictor_wrapper import PredictorWrapper
from internal.api.endpoints.camera import get_camera_router
from internal.core.camera_manager import CameraManager
from fastapi import APIRouter
from internal.utils.config_loader import AppConfig
from internal.api.endpoints.media import get_media_router
from internal.api.endpoints.predict import get_predict_router
from internal.api.endpoints.system import get_system_router

def create_router(config: AppConfig, camera_manager: CameraManager, predictor_wrapper: PredictorWrapper):
    router = APIRouter()

    # Include sub-routers
    router.include_router(get_camera_router(camera_manager))
    router.include_router(get_media_router(config))
    router.include_router(get_predict_router(config, predictor_wrapper))
    router.include_router(get_system_router())

    return router
