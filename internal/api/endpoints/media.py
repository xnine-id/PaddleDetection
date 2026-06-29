import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from internal.constants.predict_action import PredictAction
from internal.utils.config_loader import DetectionConfig
from internal.api.middleware.auth import verify_token

def get_media_router(config: DetectionConfig):
    router = APIRouter(tags=["Media"], dependencies=[Depends(verify_token)])

    snapshot_dirs = {
        PredictAction.VIDEO_ACTION: config.fight.snapshot.output_dir,
        PredictAction.VEHICLE_PLATE: config.vehicle_plate.snapshot.output_dir,
    }

    @router.get("/snapshots/{action}/{date_str}/{filename}", summary="Get snapshot image file")
    async def get_snapshot(action: PredictAction, date_str: str, filename: str):
        """
        Get snapshot image by date and filename.
        """
        if action not in snapshot_dirs:
            raise HTTPException(
                status_code=400, detail="Action not found"
            )

        # Security: Prevent directory traversal by ensuring the resolved path is within snapshot_dir
        base_dir = os.path.abspath(snapshot_dirs[action])
        requested_path = os.path.abspath(os.path.join(base_dir, date_str, filename))

        if not requested_path.startswith(base_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Snapshot not found")

        return FileResponse(requested_path, filename=filename)

    return router
