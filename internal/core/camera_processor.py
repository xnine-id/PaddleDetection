import logging
import threading
from threading import Event
from typing import Optional, Dict, Any

from internal.core.fight_detector import FightDetector
from internal.service.fight_tracker import FightTracker
from internal.service.mqtt_service import MQTTService

logger = logging.getLogger("CameraProcessor")

class CameraProcessor:
    def __init__(self,
                cam_config: Dict[str, Any],
                snapshot_config: Dict[str, Any],
                pd_config: Dict[str, Any],
                fight_detector: FightDetector,
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
        self.fight_detector = fight_detector
        self.fight_tracker = FightTracker(
            snapshot_config=snapshot_config,
            cam_name=self.cam_name, 
            mqtt_service=self.mqtt_service
        )

        pushurl = self.pd_config.get('pushurl_prefix', '')

        self.predictor = self.fight_detector.predict_livestream(self.url, pushurl, self.fight_tracker)
        self.predictor_thread: Optional[threading.Thread] = None

        if self.mqtt_service:
            self.mqtt_service.register_camera(self.cam_name, self.is_running, self._on_mqtt_command)

    def _on_mqtt_command(self, payload: Dict[str, Any]):
        """Handle incoming MQTT commands for this camera"""
        run_status: Optional[bool] = payload.get('run')
        logger.debug(f"run_status: {run_status}")
        if run_status is not None:
            if run_status:
                self.start()
            else:
                self.stop()

    def run(self):
        if not self.is_running:
            return

        self.predictor_thread = threading.Thread(target=self.predictor.run, args=(self.url, self.thread_idx), daemon=True)
        self.predictor_thread.start()

    def start(self):
        if self.is_running:
            return

        logger.info(f"[{self.cam_name}] Starting camera processor...")
        if self.predictor_thread and self.predictor_thread.is_alive():
            self.predictor_thread.join()

        self.is_running = True
        self.run()

        if self.mqtt_service:
            self.mqtt_service.publish_state(self.cam_name, self.is_running)
        logger.info(f"[{self.cam_name}] Status changed to ENABLED via MQTT")

    def stop(self):
        if not self.is_running:
            return

        logger.info(f"[{self.cam_name}] Stopping camera processor...")
        self.is_running = False
        if self.predictor_thread:
            self.predictor_thread.join()

        if self.mqtt_service:
            self.mqtt_service.publish_state(self.cam_name, self.is_running)
        logger.info(f"[{self.cam_name}] Status changed to DISABLED via MQTT")
