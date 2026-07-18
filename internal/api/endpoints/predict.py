from internal.api.middleware.auth import verify_token
from internal.api.schemas import JobCreateResponse, JobStatusResponse
from internal.constants.predict_action import PredictAction
from internal.core.predictor_wrapper import PredictorWrapper
from internal.services.base.tracker_int import TrackerInt
from internal.services.fight.video_fight_tracker import VideoFightTracker
from internal.services.vehicle_plate.images_vehicle_plate_tracker import ImagesVehiclePlateTracker
from internal.services.vehicle_plate.video_vehicle_plate_tracker import VideoVehiclePlateTracker
from internal.utils.config_loader import AppConfig

from typing import Any, Dict, List, Union
from fastapi import APIRouter, HTTPException, UploadFile, File, status, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

import logging
import mimetypes
import os
import shutil
import subprocess
import uuid

logger = logging.getLogger("API_PREDICT")

# Type alias for the two video tracker variants
_VideoTracker = Union[VideoFightTracker, VideoVehiclePlateTracker]


def get_predict_router(config: AppConfig, predictor_wrapper: PredictorWrapper):
    router = APIRouter(tags=["Predict"])

    output_dir = config.system.output_dir
    jobs: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_config_path(action: PredictAction) -> str:
        """Return the correct config path for the given action."""
        if action == PredictAction.VIDEO_ACTION:
            return config.detection.fight.config_path
        if action == PredictAction.VEHICLE_PLATE:
            return config.detection.vehicle_plate.config_path
        return config.system.config_path

    # -----------------------------------------------------------------------
    # Sync processing helpers
    # -----------------------------------------------------------------------

    def process_video_sync(action: PredictAction, temp_input: str, basename: str, tracker: TrackerInt) -> str:
        """Run video prediction and return the output filename."""
        cfg_path = _get_config_path(action)

        predictor = predictor_wrapper.predict_video(
            cfg_path=cfg_path,
            video_file=temp_input,
            output_dir=output_dir,
            trackers={action: tracker},
        )

        # Important: set_file_name to avoid NoneType error in predictor.predict_video
        predictor.set_file_name(basename)

        # Synchronous run; writes an MP4 into output_dir
        predictor.run(temp_input, thread_idx=0)

        # Determine output file path (PaddleDetection saves it as {file_name}.mp4)
        output_filename = f"{basename}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        if not os.path.exists(output_path):
            logger.error("Prediction output file not found at %s", output_path)
            raise FileNotFoundError("Prediction output file was not generated")

        # Convert to H.264 using FFmpeg (mp4v codec is not supported by browsers)
        h264_filename = f"h264_{basename}.mp4"
        h264_path = os.path.join(output_dir, h264_filename)

        logger.debug("Converting video to H.264: %s -> %s", output_path, h264_path)

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", output_path,
                "-vcodec", "libx264",
                "-acodec", "aac",
                "-pix_fmt", "yuv420p",
                "-movflags", "faststart",
                h264_path,
            ]
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output_filename = h264_filename
        except Exception as exc:
            logger.error("FFmpeg conversion failed: %s. Using original file instead.", exc)

        return output_filename

    def process_image_sync(action: PredictAction, temp_input: str, tracker: ImagesVehiclePlateTracker) -> str:
        """Run image prediction and return the output filename."""
        cfg_path = _get_config_path(action)

        predictor = predictor_wrapper.predict_image(
            cfg_path=cfg_path,
            image_file=temp_input,
            output_dir=output_dir,
            trackers={action: tracker},
        )

        # Important: set_file_name
        predictor.set_file_name(temp_input)

        # Synchronous run
        predictor.run([temp_input])

        output_filename = os.path.basename(temp_input)
        output_path = os.path.join(output_dir, output_filename)

        if not os.path.exists(output_path):
            logger.error("Prediction output file not found at %s", output_path)

        return output_filename

    # -----------------------------------------------------------------------
    # Background task wrappers
    # -----------------------------------------------------------------------

    async def bg_predict_video(
        job_id: str, action: PredictAction, temp_input: str
    ) -> None:
        jobs[job_id] = {"status": "processing"}

        try:
            tracker: _VideoTracker = (
                VideoFightTracker() if action == PredictAction.VIDEO_ACTION else VideoVehiclePlateTracker()
            )
            basename = str(uuid.uuid4())
            output_filename = await run_in_threadpool(
                process_video_sync, action, temp_input, basename, tracker
            )

            result_data: Dict[str, Any] = {
                "filename": output_filename,
                "url": f"/api/result/{output_filename}",
            }
            if action == PredictAction.VIDEO_ACTION and isinstance(tracker, VideoFightTracker):
                # Build FightDetail list from all_predictions
                details: List[Dict[str, Any]] = []
                for pred in tracker.all_predictions:
                    if pred["class"] == 1:
                        frame_ids = pred.get("frame_ids", [])
                        start_time = frame_ids[0] / 30.0 if frame_ids else 0.0
                        end_time = frame_ids[-1] / 30.0 if frame_ids else 0.0
                        details.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "score": pred["score"],
                        })
                result_data["detections"] = {
                    "fight_detected": len(tracker.get_scores()) > 0,
                    "max_score": tracker.get_highest_score(),
                    "min_score": tracker.get_lowest_score(),
                    "avg_score": tracker.get_avg_scores(),
                    "fight_frequency": tracker.get_frequency(),
                    "details": details,
                }
            elif isinstance(tracker, VideoVehiclePlateTracker):
                preds = tracker.get_all_predictions()
                total_vehicles = len(preds)
                total_plates = sum(len(p["plates"]) for p in preds)
                total_plate_scores = sum(sum(p["scores"]) for p in preds)

                avg_plate_count = total_plates / total_vehicles if total_vehicles else 0.0
                avg_vehicle_plate_score = total_plate_scores / total_plates if total_plates else 0.0
                
                result_data["detections"] = {
                    "total_vehicles": total_vehicles,
                    "avg_plate_count_per_vehicle": avg_plate_count,
                    "avg_vehicle_plate_score": avg_vehicle_plate_score,
                    "total_plates": total_plates,
                    "details": [
                        {
                            "vehicle_id": p["vehicle_id"],
                            "plates": p["plates"],
                            "scores": p["scores"],
                        }
                        for p in preds
                    ],
                }

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result_data
        except Exception as exc:
            logger.exception("Error processing video job %s: %s", job_id, exc)
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)

    async def bg_predict_image(
        job_id: str, action: PredictAction, temp_input: str
    ) -> None:
        jobs[job_id] = {"status": "processing"}

        try:
            tracker = ImagesVehiclePlateTracker()
            output_filename = await run_in_threadpool(
                process_image_sync, action, temp_input, tracker
            )

            predictions = tracker.get_all_predictions()
            details: List[Dict[str, Any]] = []

            if predictions:
                image_pred = predictions[0]
                for idx, plate in enumerate(image_pred["plates"]):
                    details.append({
                        "vehicle_id": idx + 1,
                        "plates": [plate],
                        "scores": [image_pred["scores"][idx] * 100],
                    })

            total_vehicles = len(details)
            total_plates = sum(len(d["plates"]) for d in details)
            avg_plate_count = total_plates / total_vehicles if total_vehicles else 0.0
            avg_plate_score = sum(sum(p["scores"]) for p in details) / total_plates

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "filename": output_filename,
                "url": f"/api/result/{output_filename}",
                "detections": {
                    "total_vehicles": total_vehicles,
                    "avg_plate_count_per_vehicle": avg_plate_count,
                    "avg_vehicle_plate_score": avg_plate_score,
                    "total_plates": total_plates,
                    "details": details,
                },
            }
        except Exception as exc:
            logger.exception("Error processing image job %s: %s", job_id, exc)
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @router.post(
        "/predict/{action}/video",
        summary="Run video prediction (fight detection or vehicle plate)",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_token)],
    )
    async def predict_video(
        background_tasks: BackgroundTasks, action: PredictAction, file: UploadFile = File(...)
    ):
        """Upload a video and start a background prediction job for the given action."""
        os.makedirs(output_dir, exist_ok=True)
        temp_input = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_input, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}
        background_tasks.add_task(bg_predict_video, job_id, action, temp_input)
        return {"message": "Job created", "data": {"job_id": job_id}}

    @router.post(
        "/predict/{action}/image",
        summary="Run image prediction (vehicle plate)",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_token)],
    )
    async def predict_image(
        background_tasks: BackgroundTasks, action: PredictAction, file: UploadFile = File(...)
    ):
        """Upload an image and start a background prediction job for the given action."""
        if action == PredictAction.VIDEO_ACTION:
            raise HTTPException(status_code=400, detail="Video action is not supported for image")

        os.makedirs(output_dir, exist_ok=True)
        temp_input = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_input, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}
        background_tasks.add_task(bg_predict_image, job_id, action, temp_input)
        return {"message": "Job created", "data": {"job_id": job_id}}

    @router.get(
        "/jobs/{job_id}",
        summary="Get job status",
        response_model=JobStatusResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_job_status(job_id: str):
        """Get the status of a background prediction job."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"data": {"job_id": job_id, **jobs[job_id]}}

    @router.get("/result/{filename}", summary="Get result file")
    async def get_result(filename: str):
        """Serve a generated file from the output directory."""
        if not output_dir:
            raise HTTPException(status_code=500, detail="Output directory not configured")

        base_dir = os.path.abspath(output_dir)
        requested_path = os.path.abspath(os.path.join(base_dir, filename))

        if not requested_path.startswith(base_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="File not found")

        mime_type, _ = mimetypes.guess_type(requested_path)
        return FileResponse(
            requested_path,
            media_type=mime_type or "application/octet-stream",
        )

    return router
