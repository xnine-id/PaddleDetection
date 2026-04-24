import asyncio
import logging
import threading
import time
from threading import Event
from typing import Optional, Dict, Any

from PaddleDetection.deploy.pipeline.pipeline import PipePredictor
from internal.constants.infer_name import VEHICLE_PLATE, VIDEO_ACTION
from internal.core.predictor_wrapper import PredictorWrapper
from internal.services.mqtt.base.mqtt_service_int import MQTTServiceInt
from internal.services.trackers.stream.stream_fight_tracker import StreamFightTracker
from internal.services.trackers.stream.stream_vehicle_plate_tracker import StreamVehiclePlateTracker
from internal.services.trackers.base.tracker_int import TrackerInt
from internal.utils.config_loader import DetectionConfig, SystemConfig
from internal.database.entity.camera import Camera

logger = logging.getLogger("CameraProcessor")


class CameraProcessor:
    """
    Manages the lifecycle of a camera stream, including initialization, 
    running the predictor, handling disconnects, and auto-reconnecting.
    """
    def __init__(
        self,
        cam_config: Camera,
        detection_config: DetectionConfig,
        system_config: SystemConfig,
        predictor_wrapper: PredictorWrapper,
        mqtt_services: Dict[str, MQTTServiceInt],
        thread_idx: int,
    ):
        self.cam_name = cam_config.name
        self.url = cam_config.url
        self.fight_enabled: bool = cam_config.fight_enabled
        self.vehicle_plate_enabled: bool = cam_config.vehicle_plate_enabled
        self.is_running: bool = self.fight_enabled or self.vehicle_plate_enabled

        self.system_config = system_config
        self.thread_idx = thread_idx
        self.stop_event = Event()

        self.mqtt_services = mqtt_services
        self.predictor_wrapper = predictor_wrapper
        self.detection_config = detection_config

        self.cfg_path = self._calculate_cfg_path()
        self.trackers: Dict[str, TrackerInt] = self._calculate_trackers()
        self.predictor: PipePredictor = self._init_predictor()
        self.predictor_thread: Optional[threading.Thread] = None
        self._needs_restart = False

        # Register camera with all MQTT services for command handling
        for action, service in self.mqtt_services.items():
            is_running = False
            if self.vehicle_plate_enabled and action == VEHICLE_PLATE:
                is_running = True
            elif self.fight_enabled and action == VIDEO_ACTION:
                is_running = True
            service.register_camera(
                self.cam_name, is_running, self._on_mqtt_command
            )

    def _init_predictor(self):
        predictor = self.predictor_wrapper.predict_livestream(
            cfg_path=self.cfg_path,
            cam_name=self.cam_name,
            rtsp_url=self.url,
            pushurl_prefix=self.system_config.pushurl_prefix,
            trackers=self.trackers,
        )

        return predictor

    def _calculate_trackers(self):
        trackers: Dict[str, TrackerInt] = {}
        if self.fight_enabled:
            trackers[VIDEO_ACTION] = self._init_fight_tracker()
        if self.vehicle_plate_enabled:
            trackers[VEHICLE_PLATE] = self._init_vehicle_tracker()

        return trackers

    def _init_vehicle_tracker(self):
        return StreamVehiclePlateTracker(
            snapshot_config=self.detection_config.vehicle_plate.snapshot,
            cam_name=self.cam_name,
            mqtt_service=self.mqtt_services.get(VEHICLE_PLATE),
        )

    def _init_fight_tracker(self):
        return StreamFightTracker(
            snapshot_config=self.detection_config.fight.snapshot,
            cam_name=self.cam_name,
            mqtt_service=self.mqtt_services.get(VIDEO_ACTION),
        )

    def _calculate_cfg_path(self):
        """Recalculate cfg_path based on currently enabled actions, mirroring __init__ logic."""
        if self.fight_enabled and self.vehicle_plate_enabled:
            return self.system_config.config_path
        elif self.fight_enabled:
            return self.detection_config.fight.config_path
        elif self.vehicle_plate_enabled:
            return self.detection_config.vehicle_plate.config_path

        return ""

    def _on_mqtt_command(self, topic: str, payload: Dict[str, Any]):
        """Handle incoming MQTT commands for this camera"""

        if topic.startswith(self.detection_config.fight.mqtt.command_topic_prefix):
            action = VIDEO_ACTION
        elif topic.startswith(self.detection_config.vehicle_plate.mqtt.command_topic_prefix):
            action = VEHICLE_PLATE
        else:
            logger.warning(f"[{self.cam_name}] Unknown command topic: {topic}")
            return

        run_status: Optional[bool] = payload.get("run")
        if run_status is not None:
            if run_status:
                self.resume(action)
            else:
                self.pause(action)

    def run(self):
        """
        Main loop for the camera processor.
        Continuously checks the health of the predictor thread and the camera stream.
        Handles auto-reconnect if the predictor thread dies or gets stuck (no heartbeat).
        """
        logger.info(f"[{self.cam_name}] Starting main loop for camera processor...")

        while not self.stop_event.is_set():
            if self._needs_restart:
                self._needs_restart = False
                logger.info(f"[{self.cam_name}] Restarting predictor due to configuration change...")
                if self.predictor_thread is not None:
                    self.predictor.stop()
                    if self.predictor_thread.is_alive():
                        self.predictor_thread.join(timeout=2)
                
                if self.is_running:
                    self.predictor = self._init_predictor()
                    self.predictor.last_update_time = time.time()
                    self.predictor_thread = threading.Thread(
                        target=self.predictor.run,
                        args=(self.url, self.thread_idx),
                        daemon=True,
                    )
                    self.predictor_thread.start()
                else:
                    self.predictor_thread = None

                continue

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
                        time.time() - self.predictor.last_update_time
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
                    self.predictor.last_update_time = time.time()

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
                if self.predictor_thread is not None:
                    if self.predictor_thread.is_alive():
                        self.predictor.stop()
                        self.predictor_thread.join(timeout=2)
                    self.predictor_thread = None
                time.sleep(1)

        logger.info(f"[{self.cam_name}] Main loop stopped")

    def resume(self, action: str):
        logger.info(f"[{self.cam_name}] Resuming action: {action}")

        if action == VIDEO_ACTION:
            self.fight_enabled = True
        elif action == VEHICLE_PLATE:
            self.vehicle_plate_enabled = True

        if action == VIDEO_ACTION and not action in self.trackers:
            self.trackers[action] = self._init_fight_tracker()
        elif action == VEHICLE_PLATE and not action in self.trackers:
            self.trackers[action] = self._init_vehicle_tracker()

        self.cfg_path = self._calculate_cfg_path()
        self.is_running = True

        self._needs_restart = True

        service = self.mqtt_services.get(action)
        if service:
            service.publish_state(self.cam_name, True)

        threading.Thread(
            target=lambda: asyncio.run(self._persist_action_flag(action, True)),
            daemon=True,
        ).start()

        logger.info(f"[{self.cam_name}] Action '{action}' ENABLED")

    def pause(self, action: str):
        logger.info(f"[{self.cam_name}] Pausing action: {action}")

        if action == VIDEO_ACTION:
            self.fight_enabled = False
        elif action == VEHICLE_PLATE:
            self.vehicle_plate_enabled = False

        # Reset only the affected tracker
        tracker = self.trackers.get(action)
        if tracker:
            tracker.reset()
            del self.trackers[action]

        self.cfg_path = self._calculate_cfg_path()

        # If no actions remain enabled, stop the predictor entirely
        if not self.fight_enabled and not self.vehicle_plate_enabled:
            self.is_running = False

        self._needs_restart = True

        service = self.mqtt_services.get(action)
        if service:
            service.publish_state(self.cam_name, False)

        threading.Thread(
            target=lambda: asyncio.run(self._persist_action_flag(action, False)),
            daemon=True,
        ).start()

        logger.info(f"[{self.cam_name}] Action '{action}' DISABLED")

    def stop(self):
        self.stop_event.set()
        self.predictor.stop()

        for service in self.mqtt_services.values():
            service.publish_state(self.cam_name, False)

        logger.info(f"[{self.cam_name}] Status changed to STOPPED")

    async def _persist_action_flag(self, action: str, enabled: bool):
        """Persist fight_enabled or vehicle_plate_enabled to the database."""
        from internal.database.session import get_sessionmaker
        from sqlalchemy.future import select

        field = None
        if action == VIDEO_ACTION:
            field = "fight_enabled"
        elif action == VEHICLE_PLATE:
            field = "vehicle_plate_enabled"
        else:
            return

        try:
            session_factory = get_sessionmaker()
            async with session_factory() as session:
                result = await session.execute(
                    select(Camera).where(Camera.name == self.cam_name)
                )
                camera = result.scalar_one_or_none()
                if camera:
                    setattr(camera, field, enabled)
                    await session.commit()
                    logger.info(
                        f"[{self.cam_name}] DB updated: {field}={enabled}"
                    )
                else:
                    logger.warning(
                        f"[{self.cam_name}] Camera not found in DB, skipping persist"
                    )
        except Exception as e:
            logger.exception(
                f"[{self.cam_name}] Failed to persist {field}={enabled}: {e}"
            )