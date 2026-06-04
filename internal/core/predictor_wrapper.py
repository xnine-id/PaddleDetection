from types import SimpleNamespace
from typing import Dict, Any
import paddle

from PaddleDetection.deploy.pipeline.cfg_utils import merge_cfg
from PaddleDetection.deploy.pipeline.pipeline import PipePredictor


class PredictorWrapper:
    """
    Wrapper class for the PaddleDetection Pipeline Predictor.
    Initializes and configures the underlying pipeline logic for either 
    livestream (RTSP) or static video file predictions.
    """
    def __init__(self, device: str):
        self.device = device
        print("================== paddle run check ========================")
        paddle.utils.run_check()
        print("================== paddle run check ========================")

    def predict_livestream(
        self, cfg_path: str, cam_name: str, rtsp_url: str, pushurl_prefix: str, trackers: Dict[str, Any]
    ):
        args = SimpleNamespace(
            # required
            config=cfg_path,
            # inputs (we feed RTSP directly to predictor.run)
            image_file=None,
            image_dir=None,
            video_file=None,
            video_dir=None,
            rtsp=rtsp_url,
            camera_id=-1,
            # runtime and output
            output_dir="output",
            pushurl=pushurl_prefix,
            run_mode="paddle",
            device=self.device.upper(),
            enable_mkldnn=False,
            cpu_threads=1,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            # counting/region defaults
            do_entrance_counting=False,
            do_break_in_counting=False,
            illegal_parking_time=-1,
            region_type="horizontal",
            region_polygon=[],
            secs_interval=2,
            draw_center_traj=False,
            # placeholder for -o/--opt support
            opt=None,
        )
        cfg = merge_cfg(args)
        predictor = PipePredictor(args, cfg, is_video=True, multi_camera=True)

        # Link trackers to predictor if they support it
        for name, tracker in trackers.items():
            predictor.append_tracker(name, tracker)

        filename = cam_name if cam_name else rtsp_url.split("/")[-1]
        predictor.set_file_name(filename)
        return predictor

    def predict_video(self, cfg_path: str, video_file: str, output_dir: str, trackers: Dict[str, Any]):
        args = SimpleNamespace(
            # required
            config=cfg_path,
            # inputs (we feed RTSP directly to predictor.run)
            image_file=None,
            image_dir=None,
            video_file=video_file,
            video_dir=None,
            rtsp=None,
            camera_id=-1,
            # runtime and output
            output_dir=output_dir,
            pushurl=[],
            run_mode="paddle",
            device=self.device.upper(),
            enable_mkldnn=False,
            cpu_threads=1,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            # counting/region defaults
            do_entrance_counting=False,
            do_break_in_counting=False,
            illegal_parking_time=-1,
            region_type="horizontal",
            region_polygon=[],
            secs_interval=2,
            draw_center_traj=False,
            # placeholder for -o/--opt support
            opt=None,
        )
        cfg = merge_cfg(args)
        predictor = PipePredictor(args, cfg, is_video=True)

        # Link trackers to predictor if they support it
        for name, tracker in trackers.items():
            predictor.append_tracker(name, tracker)

        filename = video_file.split("/")[-1]
        predictor.set_file_name(filename)
        return predictor

    def predict_images(self, cfg_path: str, image_dir: str, output_dir: str, trackers: Dict[str, Any]):
        args = SimpleNamespace(
            # required
            config=cfg_path,
            # inputs (we feed RTSP directly to predictor.run)
            image_file=None,
            image_dir=image_dir,
            video_file=None,
            video_dir=None,
            rtsp=None,
            camera_id=-1,
            # runtime and output
            output_dir=output_dir,
            pushurl=[],
            run_mode="paddle",
            device=self.device.upper(),
            enable_mkldnn=False,
            cpu_threads=1,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            # counting/region defaults
            do_entrance_counting=False,
            do_break_in_counting=False,
            illegal_parking_time=-1,
            region_type="horizontal",
            region_polygon=[],
            secs_interval=2,
            draw_center_traj=False,
            # placeholder for -o/--opt support
            opt=None,
        )
        cfg = merge_cfg(args)
        predictor = PipePredictor(args, cfg, is_video=False)

        # Link trackers to predictor if they support it
        for name, tracker in trackers.items():
            predictor.append_tracker(name, tracker)

        return predictor

    def predict_image(self, cfg_path: str, image_file: str, output_dir: str, trackers: Dict[str, Any]):
        args = SimpleNamespace(
            # required
            config=cfg_path,
            # inputs
            image_file=image_file,
            image_dir=None,
            video_file=None,
            video_dir=None,
            rtsp=None,
            camera_id=-1,
            # runtime and output
            output_dir=output_dir,
            pushurl=[],
            run_mode="paddle",
            device=self.device.upper(),
            enable_mkldnn=False,
            cpu_threads=1,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            # counting/region defaults
            do_entrance_counting=False,
            do_break_in_counting=False,
            illegal_parking_time=-1,
            region_type="horizontal",
            region_polygon=[],
            secs_interval=2,
            draw_center_traj=False,
            # placeholder for -o/--opt support
            opt=None,
        )
        cfg = merge_cfg(args)
        predictor = PipePredictor(args, cfg, is_video=False)

        # Link trackers to predictor if they support it
        for name, tracker in trackers.items():
            predictor.append_tracker(name, tracker)

        predictor.set_file_name(image_file)
        return predictor
