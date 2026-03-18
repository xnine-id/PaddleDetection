import logging
import threading
import time
from threading import Event
from typing import Optional, Dict, Any

from internal.core.fight_detector import FightDetector
from internal.services.stream_fight_tracker import StreamFightTracker
from internal.services.mqtt_service import MQTTService

logger = logging.getLogger("CameraProcessor")


class CameraProcessor:
    def __init__(
        self,
        cam_config: Dict[str, Any],
        snapshot_config: Dict[str, Any],
        pd_config: Dict[str, Any],
        fight_detector: FightDetector,
        mqtt_service: Optional[MQTTService],
        thread_idx: int,
        stop_event: Event,
    ):
        self.cam_name = cam_config["name"]
        self.url = cam_config["url"]
        self.is_running: bool = cam_config["enabled"]
        self.pd_config = pd_config or {}
        self.thread_idx = thread_idx
        self.stop_event = stop_event

        self.mqtt_service = mqtt_service
        self.fight_detector = fight_detector
        self.fight_tracker = StreamFightTracker(
            snapshot_config=snapshot_config,
            cam_name=self.cam_name,
            mqtt_service=self.mqtt_service,
        )

        pushurl = self.pd_config.get("pushurl_prefix", "")

        self.predictor = self.fight_detector.predict_livestream(
            self.cam_name, self.url, pushurl, self.fight_tracker
        )
        self.predictor_thread: Optional[threading.Thread] = None

        if self.mqtt_service:
            self.mqtt_service.register_camera(
                self.cam_name, self.is_running, self._on_mqtt_command
            )

    def _on_mqtt_command(self, payload: Dict[str, Any]):
        """Handle incoming MQTT commands for this camera"""
        run_status: Optional[bool] = payload.get("run")
        logger.debug(f"run_status: {run_status}")
        if run_status is not None:
            if run_status:
                self.start()
            else:
                self.stop()

    def run(self):
        """Main loop for camera processor, handles auto-reconnect and stuck detection"""
        logger.info(f"[{self.cam_name}] Starting main loop for camera processor...")

        while not self.stop_event.is_set():
            if self.is_running:
                # 1. Check if predictor thread is dead
                thread_is_alive = (
                    self.predictor_thread is not None
                    and self.predictor_thread.is_alive()
                )

                # 2. Check if predictor is stuck (no heartbeat for > 15 seconds)
                is_stuck = False
                if thread_is_alive:
                    time_since_last_update = (
                        time.time() - self.fight_tracker.last_update_time
                    )
                    if time_since_last_update > 15:
                        logger.warning(
                            f"[{self.cam_name}] Predictor seems stuck (no heartbeat for {time_since_last_update:.1f}s)"
                        )
                        is_stuck = True

                # Reconnect if dead or stuck
                if not thread_is_alive or is_stuck:
                    if self.predictor_thread is not None:
                        reason = "died" if not thread_is_alive else "stuck"
                        logger.warning(
                            f"[{self.cam_name}] Predictor {reason}. Reconnecting..."
                        )

                        # Stop existing predictor and its capture before starting a new one
                        # This prevents thread and resource accumulation (memory leak)
                        logger.info(
                            f"[{self.cam_name}] Stopping existing predictor resources..."
                        )
                        self.predictor.stop()

                        # Wait a bit for the thread to exit if it's not totally hung
                        if self.predictor_thread.is_alive():
                            self.predictor_thread.join(timeout=2)

                    logger.info(f"[{self.cam_name}] Connecting to camera: {self.url}")
                    # Reset heartbeat before starting
                    self.fight_tracker.heartbeat()

                    self.predictor_thread = threading.Thread(
                        target=self.predictor.run,
                        args=(self.url, self.thread_idx),
                        daemon=True,
                    )
                    self.predictor_thread.start()

                # Sleep a bit before checking again
                time.sleep(2)
            else:
                # If we are not supposed to be running, but thread is still alive,
                # we just wait for it to die (it should die if stream is closed or predictor returns)
                # Note: We can't easily force-kill a thread in Python
                time.sleep(1)

        logger.info(f"[{self.cam_name}] Main loop stopped")

    def start(self):
        if self.is_running:
            return

        logger.info(f"[{self.cam_name}] Enabling camera processor status...")
        self.is_running = True

        if self.mqtt_service:
            self.mqtt_service.publish_state(self.cam_name, self.is_running)
        logger.info(f"[{self.cam_name}] Status changed to ENABLED")

    def stop(self):
        if not self.is_running:
            return

        logger.info(f"[{self.cam_name}] Disabling camera processor status...")
        self.is_running = False

        # Stop predictor resources
        self.predictor.stop()

        # Reset fight tracker
        self.fight_tracker.reset()

        # Note: predictor_thread will continue until its current run() call finishes.
        # This usually happens when the stream is closed or an error occurs.
        # We don't join here because it might block MQTT/API response if the stream is hanging.

        if self.mqtt_service:
            self.mqtt_service.publish_state(self.cam_name, self.is_running)
        logger.info(f"[{self.cam_name}] Status changed to DISABLED")
