from internal.api.middleware.auth import verify_token
from fastapi.param_functions import Depends
from fastapi.responses import FileResponse
import mimetypes
import logging
import os
import shutil
import uuid
import subprocess
from fastapi import APIRouter, HTTPException, UploadFile, File, status, BackgroundTasks
from starlette.concurrency import run_in_threadpool

from internal.constants.infer_name import VEHICLE_PLATE, VIDEO_ACTION
from internal.core.predictor_wrapper import PredictorWrapper
from internal.services.vehicle_plate.images_vehicle_plate_tracker import ImagesVehiclePlateTracker
from internal.services.vehicle_plate.video_vehicle_plate_tracker import VideoVehiclePlateTracker
from internal.services.fight.video_fight_tracker import VideoFightTracker
from internal.utils.config_loader import AppConfig
from internal.api.schemas import JobCreateResponse, JobStatusResponse

logger = logging.getLogger("API_PREDICT")

def get_predict_router(config: AppConfig, predictor_wrapper: PredictorWrapper):
    router = APIRouter(tags=["Predict"])

    output_dir = config.system.output_dir
    jobs = {}

    async def _process_video_prediction(action: str, temp_input: str, original_filename: str, tracker):
        """
        Helper to handle common video upload, prediction run, and FFmpeg conversion.
        """
        if not output_dir:
            raise HTTPException(
                status_code=500, detail="Output directory not configured"
            )

        basename = str(uuid.uuid4())

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        def _run_pred():
            # Run prediction
            cfg_path = config.system.config_path
            if action == VIDEO_ACTION:
                cfg_path = config.detection.fight.config_path
            elif action == VEHICLE_PLATE:
                cfg_path = config.detection.vehicle_plate.config_path

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
        
        await run_in_threadpool(_run_pred)

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
            await run_in_threadpool(
                subprocess.run,
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

    async def _process_image_prediction(action: str, temp_input: str, original_filename: str, tracker):
        """
        Helper to handle common image upload, prediction run.
        """
        if not output_dir:
            raise HTTPException(
                status_code=500, detail="Output directory not configured"
            )
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        def _run_pred():
            # Run prediction
            cfg_path = config.system.config_path
            if action == VIDEO_ACTION:
                cfg_path = config.detection.fight.config_path
            elif action == VEHICLE_PLATE:
                cfg_path = config.detection.vehicle_plate.config_path

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
            
        await run_in_threadpool(_run_pred)

        # Determine output file path
        output_filename = os.path.basename(temp_input)
        output_path = os.path.join(output_dir, output_filename)

        if not os.path.exists(output_path):
            logger.error(f"Prediction output file not found at {output_path}")

        return output_filename

    async def bg_predict_video(job_id: str, action: str, temp_input: str, original_filename: str):
        jobs[job_id] = {"status": "processing"}

        try:
            tracker = VideoFightTracker() if action == VIDEO_ACTION else VideoVehiclePlateTracker()
            output_filename = await _process_video_prediction(action, temp_input, original_filename, tracker)

            result_data = {"filename": output_filename, "url": f"/api/result/{output_filename}"}
            if action == VIDEO_ACTION:
                result_data["detections"] = {
                    'fight_detected': len(tracker.get_scores()) > 0,
                    "max_score": tracker.get_highest_score(),
                    "min_score": tracker.get_lowest_score(),
                    "avg_score": tracker.get_avg_scores(),
                    "fight_frequency": tracker.get_frequency(),
                }
            else:
                result_data["detections"] = tracker.get_all_predictions()

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result_data
        except Exception as e:
            logger.exception(f"Error processing video job {job_id}: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)

    async def bg_predict_image(job_id: str, action: str, temp_input: str, original_filename: str):
        jobs[job_id] = {"status": "processing"}
        try:
            tracker = ImagesVehiclePlateTracker()
            output_filename = await _process_image_prediction(action, temp_input, original_filename, tracker)

            predictions = tracker.get_all_predictions()
            result = []

            if len(predictions) > 0:
                for idx, plate in enumerate(predictions[0]['plates']):
                    result.append({
                        "vehicle_id": idx+1,
                        "plates": [plate],
                        "scores": [predictions[0]['scores'][idx] * 100]
                    })

            result_data = {
                "filename": output_filename, 
                "url": f"/api/result/{output_filename}",
                "detections": result,
            }
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result_data
        except Exception as e:
            logger.exception(f"Error processing image job {job_id}: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)

    @router.post(
        "/predict/video_action/video",
        summary="Predict fight from uploaded video",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_token)]
    )
    async def predict_fight_from_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        """
        Upload a video file, start background job for fight detection, and return job id.
        """
        temp_input = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}

        background_tasks.add_task(bg_predict_video, job_id, VIDEO_ACTION, temp_input, file.filename)

        return {"message": "Job created", "data": {"job_id": job_id}}

    @router.post(
        "/predict/vehicleplate/video",
        summary="Predict vehicle plates from uploaded video",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_token)]
    )
    async def predict_vehicle_plate_from_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        """
        Upload a video file, start background job for vehicle plate detection, and return job id.
        """
        temp_input = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}

        background_tasks.add_task(bg_predict_video, job_id, VEHICLE_PLATE, temp_input, file.filename)

        return {"message": "Job created", "data": {"job_id": job_id}}

    @router.post(
        "/predict/vehicleplate/image",
        summary="Predict vehicle plates from uploaded image",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_token)]
    )
    async def predict_vehicle_plate_from_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        """
        Upload an image file, start background job for vehicle plate detection, and return job id.
        """
        temp_input = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}

        background_tasks.add_task(bg_predict_image, job_id, VEHICLE_PLATE, temp_input, file.filename)

        return {"message": "Job created", "data": {"job_id": job_id}}

    @router.get("/jobs/{job_id}", summary="Get job status", response_model=JobStatusResponse, dependencies=[Depends(verify_token)])
    async def get_job_status(job_id: str):
        """
        Get the status of a background prediction job.
        """
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {"data": {"job_id": job_id, **jobs[job_id]}}

    @router.get("/result/{filename}", summary="Get result file")
    async def get_result(filename: str):
        """
        Serve a generated file from the output directory.
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
            raise HTTPException(status_code=404, detail="File not found")

        mime_type, _ = mimetypes.guess_type(requested_path)
        return FileResponse(requested_path, media_type=mime_type or "application/octet-stream", filename=filename)

    return router
