import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from internal.api.middleware.auth import verify_token
from internal.utils.config_loader import AppConfig
from internal.constants.infer_name import VEHICLE_PLATE, VIDEO_ACTION

def get_media_router(config: AppConfig):
    router = APIRouter(tags=["Media"])

    snapshot_dirs = {
        VIDEO_ACTION: config.detection.fight.snapshot.output_dir,
        VEHICLE_PLATE: config.detection.vehicle_plate.snapshot.output_dir,
    }

    @router.get(
        "/snapshots/{action}/{date_str}/{filename}",
        summary="Get snapshot image file",
        dependencies=[Depends(verify_token)]
    )
    async def get_snapshot(action: str, date_str: str, filename: str):
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
