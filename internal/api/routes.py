import logging
import os
import tempfile
import shutil
import uuid
import subprocess
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse, JSONResponse
from internal.core.fight_detector import FightDetector
from internal.service.video_fight_tracker import VideoFightTracker

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
        if not output_dir:
            raise HTTPException(status_code=500, detail="Output directory not configured")

        work_dir = tempfile.mkdtemp(prefix="fight_pred_")
        try:
            # Save uploaded file to a temporary location
            ext = os.path.splitext(file.filename or "uploaded.mp4")[1] or ".mp4"
            basename = str(uuid.uuid4())

            input_path = os.path.join(work_dir, f"{basename}{ext}")
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            fight_tracker = VideoFightTracker()

            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Run prediction
            predictor = fight_detector.predict_video(
                video_file=input_path,
                output_dir=output_dir,
                fight_tracker=fight_tracker,
            )
            
            # Important: set_file_name to avoid NoneType error in predictor.predict_video
            predictor.set_file_name(basename)
            
            # Synchronous run; writes an MP4 into output_dir
            predictor.run(input_path, thread_idx=0)

            # Determine output file path (PaddleDetection saves it as {file_name}.mp4)
            output_filename = f"{basename}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            if not os.path.exists(output_path):
                logger.error(f"Prediction output file not found at {output_path}")
                raise Exception("Prediction output file was not generated")

            # 3.1 Convert to H.264 using FFmpeg (mp4v codec is not supported by browsers)
            h264_filename = f"h264_{basename}.mp4"
            h264_path = os.path.join(output_dir, h264_filename)

            logger.debug(f"Converting video to H.264: {output_path} -> {h264_path}")

            try:
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", output_path,
                    "-vcodec", "libx264", "-acodec", "aac",
                    "-pix_fmt", "yuv420p", "-movflags", "faststart",
                    h264_path
                ]
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # If conversion success, use the h264 file
                output_filename = h264_filename
            except Exception as e:
                logger.error(f"FFmpeg conversion failed: {str(e)}. Using original file instead.")
                # If conversion fails, we'll try to proceed with the original file

            # Compute avg score from fight tracker
            avg_score = fight_tracker.get_avg_scores()

            url = f"/api/videos/{output_filename}"
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"data": {"url": url, "score": avg_score}}
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Prediction failed: {str(e)}"}
            )
        finally:
            # Cleanup temp work dir
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {work_dir}: {e}")

    return router
