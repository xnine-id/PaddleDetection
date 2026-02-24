import logging
import threading
import time
from threading import Event
from typing import Optional, Dict, Any
from types import SimpleNamespace

from PaddleDetection.deploy.pipeline.pipeline import PipePredictor
from PaddleDetection.deploy.pipeline.cfg_utils import merge_cfg
from internal.service.fight_tracker import FightTracker
from internal.service.mqtt_service import MQTTService

logger = logging.getLogger("CameraProcessor")

class CameraProcessor:
    def __init__(self,
                cam_config: Dict[str, Any],
                snapshot_config: Dict[str, Any],
                pd_config: Dict[str, Any],
                mqtt_service: Optional[MQTTService],
                thread_idx: int,
                stop_event: Event):
        self.cam_name = cam_config['name']
        self.url = cam_config['url']
        self.is_running: bool = cam_config['enabled']
        self.pd_config = pd_config or {}
        self.thread_idx = thread_idx
        self.stop_event = stop_event

        self.mqtt_service = mqtt_service
        self.fight_tracker = FightTracker(
            snapshot_config=snapshot_config,
            cam_name=self.cam_name, 
            mqtt_service=self.mqtt_service
        )

        # Build args and cfg for PipePredictor
        cfg_path = self.pd_config.get('config_path')
        device = str(self.pd_config.get('device', 'cpu')).upper()
        pushurl = self.pd_config.get('pushurl_prefix', '')

        args = SimpleNamespace(
            # required
            config=cfg_path,
            # inputs (we feed RTSP directly to predictor.run)
            image_file=None,
            image_dir=None,
            video_file=None,
            video_dir=None,
            rtsp=self.url,
            camera_id=-1,
            # runtime and output
            output_dir="output",
            pushurl=pushurl,
            run_mode='paddle',
            device=device,
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
            region_type='horizontal',
            region_polygon=[],
            secs_interval=2,
            draw_center_traj=False,
            # placeholder for -o/--opt support
            opt=None
        )
        cfg = merge_cfg(args)
        self.predictor = PipePredictor(args, cfg, is_video=True, multi_camera=True)
        self.predictor.set_file_name(self.cam_name)

        if self.mqtt_service:
            self.mqtt_service.register_camera(self.cam_name, self.is_running, self._on_mqtt_command)

    def _on_mqtt_command(self, payload: Dict[str, Any]):
        """Handle incoming MQTT commands for this camera"""
        run_status: Optional[bool] = payload.get('run')
        if run_status is not None:
            self.is_running = run_status
            if self.mqtt_service:
                self.mqtt_service.publish_state(self.cam_name, self.is_running)
            status_str = "ENABLED" if run_status else "DISABLED"
            logger.info(f"[{self.cam_name}] Status changed to {status_str} via MQTT")

    def run(self):
        track_thread = threading.Thread(target=self.track_fight_result, daemon=True)
        track_thread.start()

        self.predictor.run(self.url, self.thread_idx)

    def track_fight_result(self):
        while not self.stop_event.is_set():
            if self.is_running:
                try:
                    self.fight_tracker.update(self.predictor.pipeline_res)
                except Exception as e:
                    logger.error(f"Error on update fight tracker: {e}")
                time.sleep(0.05)
            else:
                time.sleep(0.2)
