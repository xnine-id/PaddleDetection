from internal.services.trackers.images.images_vehicle_plate_tracker import ImagesVehiclePlateTracker
from internal.services.trackers.video.video_vehicle_plate_tracker import VideoVehiclePlateTracker
import logging
import os
import tempfile
import shutil
import uuid
import subprocess
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse, JSONResponse
from internal.constants.infer_name import VEHICLE_PLATE, VIDEO_ACTION
from internal.core.predictor_wrapper import PredictorWrapper
from internal.services.trackers.video.video_fight_tracker import VideoFightTracker

from internal.utils.config_loader import AppConfig

logger = logging.getLogger("API_ROUTES")


def create_router(config: AppConfig):
    router = APIRouter()

    snapshot_dir = {
        VIDEO_ACTION: config.detection.fight.snapshot.output_dir,
        VEHICLE_PLATE: config.detection.vehicle_plate.snapshot.output_dir,
    }
    predictor_wrapper = {
        VIDEO_ACTION: PredictorWrapper(config.detection.fight.config_path, device=config.system.device),
        VEHICLE_PLATE: PredictorWrapper(config.detection.vehicle_plate.config_path, device=config.system.device),
    }
    output_dir = config.system.output_dir

    @router.get(
        "/snapshots/{action}/{date_str}/{filename}",
        summary="Get snapshot image file",
        tags=["Snapshot"],
    )
    async def get_snapshot(action: str, date_str: str, filename: str):
        """
        Get snapshot image by date and filename.
        """
        if action not in snapshot_dir:
            raise HTTPException(
                status_code=400, detail="Action not found"
            )

        # Security: Prevent directory traversal by ensuring the resolved path is within snapshot_dir
        base_dir = os.path.abspath(snapshot_dir[action])
        requested_path = os.path.abspath(os.path.join(base_dir, date_str, filename))

        if not requested_path.startswith(base_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Snapshot not found")

        return FileResponse(requested_path, filename=filename)

    @router.get("/videos/{filename}", summary="Get result video file", tags=["Video"])
    async def get_video(filename: str):
        """
        Serve a generated video file from the output directory.
        """
        if not output_dir:
            raise HTTPException(
                status_code=500, detail="Output directory not configured"
            )
        base_dir = os.path.abspath(output_dir)
        requested_path = os.path.abspath(os.path.join(base_dir, filename))

        if not requested_path.startswith(base_dir):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Video not found")

        import mimetypes
        mime_type, _ = mimetypes.guess_type(requested_path)
        return FileResponse(requested_path, media_type=mime_type or "application/octet-stream", filename=filename)

    async def _process_video_prediction(action: str, file: UploadFile, tracker):
        """
        Helper to handle common video upload, prediction run, and FFmpeg conversion.
        """
        if not output_dir:
            raise HTTPException(
                status_code=500, detail="Output directory not configured"
            )

        if action not in predictor_wrapper:
            raise HTTPException(
                status_code=400, detail=f"Predictor for action '{action}' not found"
            )

        work_dir = tempfile.mkdtemp(prefix=f"{action}_pred_")
        try:
            # Save uploaded file to a temporary location
            ext = os.path.splitext(file.filename or "uploaded.mp4")[1] or ".mp4"
            basename = str(uuid.uuid4())

            input_path = os.path.join(work_dir, f"{basename}{ext}")
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Run prediction
            predictor = predictor_wrapper[action].predict_video(
                video_file=input_path,
                output_dir=output_dir,
                trackers={action: tracker},
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

            # Convert to H.264 using FFmpeg (mp4v codec is not supported by browsers)
            h264_filename = f"h264_{basename}.mp4"
            h264_path = os.path.join(output_dir, h264_filename)

            logger.debug(f"Converting video to H.264: {output_path} -> {h264_path}")

            try:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    output_path,
                    "-vcodec",
                    "libx264",
                    "-acodec",
                    "aac",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "faststart",
                    h264_path,
                ]
                subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                output_filename = h264_filename
            except Exception as e:
                logger.error(
                    f"FFmpeg conversion failed: {str(e)}. Using original file instead."
                )

            return output_filename
        finally:
            # Cleanup temp work dir
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {work_dir}: {e}")

    async def _process_image_prediction(action: str, file: UploadFile, tracker):
        """
        Helper to handle common image upload, prediction run.
        """
        if not output_dir:
            raise HTTPException(
                status_code=500, detail="Output directory not configured"
            )

        if action not in predictor_wrapper:
            raise HTTPException(
                status_code=400, detail=f"Predictor for action '{action}' not found"
            )

        work_dir = tempfile.mkdtemp(prefix=f"{action}_img_pred_")
        try:
            # Save uploaded file to a temporary location
            ext = os.path.splitext(file.filename or "uploaded.jpg")[1] or ".jpg"
            basename = str(uuid.uuid4())

            input_path = os.path.join(work_dir, f"{basename}{ext}")
            with open(input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Run prediction
            predictor = predictor_wrapper[action].predict_image(
                image_file=input_path,
                output_dir=output_dir,
                trackers={action: tracker},
            )

            # Important: set_file_name
            predictor.set_file_name(input_path)

            # Synchronous run
            predictor.run([input_path])

            # Determine output file path (basename remains same, in output_dir)
            output_filename = f"{basename}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            if not os.path.exists(output_path):
                logger.error(f"Prediction output file not found at {output_path}")
                raise Exception("Prediction output file was not generated")

            return output_filename
        finally:
            # Cleanup temp work dir
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {work_dir}: {e}")

    @router.post(
        "/predict/video_action/video", summary="Predict fight from uploaded video", tags=["Predict"]
    )
    async def predict_fight_from_upload(file: UploadFile = File(...)):
        """
        Upload a video file and run fight detection. Returns a JSON with URL and avg score.
        """
        try:
            tracker = VideoFightTracker()
            output_filename = await _process_video_prediction(VIDEO_ACTION, file, tracker)

            avg_score = tracker.get_avg_scores()
            url = f"/api/videos/{output_filename}"

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"data": {"filename": output_filename, "url": url, "avg_score": avg_score}},
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Fight prediction failed: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Prediction failed: {str(e)}"},
            )

    @router.post(
        "/predict/vehicleplate/video", summary="Predict vehicle plates from uploaded video", tags=["Predict"]
    )
    async def predict_vehicle_plate_from_upload(file: UploadFile = File(...)):
        """
        Upload a video file and run vehicle plate detection. Returns a JSON with URL and detections.
        """
        try:
            tracker = VideoVehiclePlateTracker()
            output_filename = await _process_video_prediction(VEHICLE_PLATE, file, tracker)

            predictions = tracker.get_all_predictions()
            url = f"/api/videos/{output_filename}"

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"data": {"filename": output_filename, "url": url, "detections": predictions}},
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vehicle plate prediction failed: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Prediction failed: {str(e)}"},
            )

    @router.post(
        "/predict/vehicleplate/image", summary="Predict vehicle plates from uploaded image", tags=["Predict"]
    )
    async def predict_vehicle_plate_from_image(file: UploadFile = File(...)):
        """
        Upload a image file and run vehicle plate detection. Returns a JSON with URL and detections.
        """
        try:
            tracker = ImagesVehiclePlateTracker()
            output_filename = await _process_image_prediction(VEHICLE_PLATE, file, tracker)

            predictions = tracker.get_all_predictions()
            url = f"/api/videos/{output_filename}"

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"data": {"filename": output_filename, "url": url, "detections": predictions[0] if len(predictions) >= 1 else None}},
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vehicle plate prediction failed: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Prediction failed: {str(e)}"},
            )


    return router
