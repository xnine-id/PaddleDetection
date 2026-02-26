import logging
import os
import tempfile
import shutil
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse, JSONResponse
from internal.core.fight_detector import FightDetector
from internal.service.api_fight_tracker import ApiFightTracker

logger = logging.getLogger("API_ROUTES")

def create_router(fight_detector: FightDetector, config: dict = None):
    router = APIRouter()

    snapshot_dir = config.get('snapshot', {}).get('output_dir') if config else None
    output_dir = config.get('paddle_detection', {}).get('output_dir') if config else None

    @router.get("/videos/{filename}", summary="Get result video file", tags=["Video"])
    async def get_video(filename: str):
        if not output_dir:
            raise HTTPException(status_code=500, detail="Output directory not configured")
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(file_path, media_type="video/mp4", filename=filename)

    @router.get("/snapshots/{date_str}/{filename}", summary="Get snapshot image file", tags=["Snapshot"])
    async def get_snapshot(date_str: str, filename: str):
        """
        Get snapshot image by date and filename.
        """
        if not snapshot_dir:
            raise HTTPException(status_code=500, detail="Snapshot directory not configured")
        
        file_path = os.path.join(snapshot_dir, date_str, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Snapshot not found")
            
        return FileResponse(file_path)

    @router.post("/predict/video", summary="Predict from uploaded video", tags=["Predict"])
    async def predict_from_upload(
        file: UploadFile = File(...)
    ):
        """
        Upload a video file and run fight detection. Returns a JSON with URL and avg score.
        """
        try:
            # Save uploaded file to a temporary location
            ext = os.path.splitext(file.filename or "uploaded.mp4")[1] or ".mp4"
            basename = str(uuid.uuid4())

            work_dir = tempfile.mkdtemp(prefix="fight_pred_")
            input_path = os.path.join(work_dir, f"{basename}{ext}")
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            fight_tracker = ApiFightTracker()

            # Run prediction
            predictor = fight_detector.predict_video(
                video_file=input_path,
                output_dir=output_dir,
                fight_tracker=fight_tracker,
            )
            # Synchronous run; writes an MP4 into output_dir
            predictor.run(input_path, thread_idx=0)

            # Determine output file path
            base_name = os.path.splitext(os.path.basename(input_path))[0]

            # Compute avg score from fight tracker
            avg_score = fight_tracker.get_avg_scores()

            url = f"/api/videos/{base_name}{ext}"
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"url": url, "score": avg_score}
            )
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Prediction failed: {str(e)}"}
            )

    return router
