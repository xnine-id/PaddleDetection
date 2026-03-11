from types import SimpleNamespace

from PaddleDetection.deploy.pipeline.cfg_utils import merge_cfg
from PaddleDetection.deploy.pipeline.pipeline import PipePredictor
from internal.service.fight_tracker_int import FightTrackerInt


class FightDetector:
    def __init__(self, cfg_path: str, device: str):
        self.cfg_path = cfg_path
        self.device = device

    def predict_livestream(
        self, cam_name: str, rtsp_url: str, pushurl: str, fight_tracker: FightTrackerInt
    ):
        args = SimpleNamespace(
            # required
            config=self.cfg_path,
            # inputs (we feed RTSP directly to predictor.run)
            image_file=None,
            image_dir=None,
            video_file=None,
            video_dir=None,
            rtsp=rtsp_url,
            camera_id=-1,
            # runtime and output
            output_dir='output',
            pushurl=pushurl,
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
        predictor = PipePredictor(
            args, cfg, is_video=True, multi_camera=True, fight_tracker=fight_tracker
        )

        filename = cam_name if cam_name else rtsp_url.split("/")[-1]
        predictor.set_file_name(filename)
        return predictor

    def predict_video(
        self, video_file: str, output_dir: str, fight_tracker: FightTrackerInt
    ):
        args = SimpleNamespace(
            # required
            config=self.cfg_path,
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
        predictor = PipePredictor(args, cfg, is_video=True, fight_tracker=fight_tracker)

        filename = video_file.split("/")[-1]
        predictor.set_file_name(filename)
        return predictor
