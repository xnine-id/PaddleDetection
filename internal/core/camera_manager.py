from typing import Any, Optional
import asyncio
import threading
from typing import Dict, List
import logging
from sqlalchemy.future import select
from internal.core.predictor_wrapper import PredictorWrapper
from internal.core.camera_processor import CameraProcessor
from internal.services.base.mqtt_service_int import MQTTServiceInt
from internal.services.fight.fight_mqtt_service import FightMQTTService
from internal.services.vehicle_plate.plate_mqtt_service import PlateMQTTService
from internal.utils.config_loader import AppConfig
from internal.constants.predict_action import PredictAction
from internal.database.entity.camera import Camera
from internal.database.session import get_sessionmaker

logger = logging.getLogger("CameraManager")


class CameraManager:
    """
    Manages the lifecycle and streaming of multiple camera processors.

    This class is responsible for initializing ``CameraProcessor`` instances
    for each configured camera, starting them in separate threads, and
    routing MQTT commands to the correct processor.
    """
    def __init__(self, predictor_wrapper: PredictorWrapper, config: AppConfig):
        self.config = config
        self.predictor_wrapper = predictor_wrapper
        
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        self.camera_processors: Dict[str, CameraProcessor] = {}
        self.mqtt_services: Dict[str, MQTTServiceInt] = {}
        self._lock = threading.Lock()
        
        self._init_mqtt_services()


    def _init_mqtt_services(self) -> None:
        # Initialize MQTT services for each detection module if enabled
        if self.config.detection.fight.mqtt.enabled:
            self.mqtt_services[PredictAction.VIDEO_ACTION] = FightMQTTService(
                self.config.detection.fight.mqtt, self._on_mqtt_command
            )

        if self.config.detection.vehicle_plate.mqtt.enabled:
            self.mqtt_services[PredictAction.VEHICLE_PLATE] = PlateMQTTService(
                self.config.detection.vehicle_plate.mqtt, self._on_mqtt_command
            )

    # ------------------------------------------------------------------
    # MQTT command handling
    # ------------------------------------------------------------------

    def _on_mqtt_command(self, cam_name: str, topic: str, payload: Dict[str, Any]) -> None:
        """
        Entry point for all inbound MQTT commands.

        Resolves which action and run-state the message describes, then
        delegates to ``_handle_resume`` or ``_handle_pause``.
        """
        if topic.startswith(self.config.detection.fight.mqtt.command_topic_prefix):
            action = PredictAction.VIDEO_ACTION
        elif topic.startswith(self.config.detection.vehicle_plate.mqtt.command_topic_prefix):
            action = PredictAction.VEHICLE_PLATE
        else:
            logger.warning("[%s] Unknown command topic: %s", cam_name, topic)
            return

        run: Optional[bool] = payload.get("run")
        if run is None:
            logger.warning("[%s] Missing 'run' key in payload: %s", cam_name, payload)
            return

        self._persist_enabled_async(cam_name, action, run)

        processor = self._get_processor(cam_name)

        if run:
            self._handle_resume(cam_name, action, processor)
        else:
            self._handle_pause(cam_name, action, processor)

    def _handle_resume(
        self,
        cam_name: str,
        action: PredictAction,
        processor: Optional[CameraProcessor],
    ) -> None:
        """Resume *action* on the processor, creating it first if needed."""
        if processor is None:
            # Processor doesn't exist yet — boot it from DB then resume inside add_camera_processor
            threading.Thread(
                target=lambda: asyncio.run(
                    self._boot_processor_from_db(cam_name, action)
                ),
                daemon=True,
            ).start()
            return

        processor.resume(action)

    def _handle_pause(
        self,
        cam_name: str,
        action: PredictAction,
        processor: Optional[CameraProcessor],
    ) -> None:
        """Pause *action* on the processor; warn if it no longer exists."""
        if processor is None:
            logger.warning("[%s] Cannot pause — processor not found.", cam_name)
            return

        processor.pause(action)

    # ------------------------------------------------------------------
    # DB persistence (fire-and-forget)
    # ------------------------------------------------------------------

    def _persist_enabled_async(
        self, cam_name: str, action: PredictAction, enabled: bool
    ) -> None:
        """Spawn a daemon thread that writes *enabled* to the database."""
        threading.Thread(
            target=lambda: asyncio.run(self._update_db_enabled(cam_name, action, enabled)),
            daemon=True,
        ).start()

    async def _update_db_enabled(
        self, cam_name: str, action: PredictAction, enabled: bool
    ) -> None:
        field_map = {
            PredictAction.VIDEO_ACTION: "fight_enabled",
            PredictAction.VEHICLE_PLATE: "vehicle_plate_enabled",
        }
        field = field_map.get(action)
        if field is None:
            logger.warning("[%s] Unknown action for DB update: %s", cam_name, action)
            return

        try:
            session_factory = get_sessionmaker()
            async with session_factory() as session:
                result = await session.execute(
                    select(Camera).where(Camera.name == cam_name)
                )
                camera = result.scalar_one_or_none()
                if camera is None:
                    logger.warning("[%s] Camera not found in DB.", cam_name)
                    return
                setattr(camera, field, enabled)
                await session.commit()
                logger.debug("[%s] DB updated: %s=%s", cam_name, field, enabled)
        except Exception:
            logger.exception("[%s] Failed to persist %s=%s to DB.", cam_name, field, enabled)

    async def _boot_processor_from_db(
        self, cam_name: str, action: Optional[PredictAction] = None
    ) -> None:
        """Load a camera record from the DB and start a new processor for it.

        When *action* is provided, the corresponding enabled flag is forced to
        ``True`` on the camera object **before** the processor is created.
        This avoids a race with the fire-and-forget DB update that may not
        have been committed yet.
        """
        try:
            session_factory = get_sessionmaker()
            async with session_factory() as session:
                result = await session.execute(
                    select(Camera).where(Camera.name == cam_name)
                )
                camera = result.scalar_one_or_none()

            if camera is None:
                logger.warning("[%s] Camera not found in DB — cannot start processor.", cam_name)
                return

            # Override enabled flag from the MQTT intent (don't trust the
            # possibly-stale DB row that the fire-and-forget write hasn't
            # flushed yet).
            if action is not None:
                field_map = {
                    PredictAction.VIDEO_ACTION: "fight_enabled",
                    PredictAction.VEHICLE_PLATE: "vehicle_plate_enabled",
                }
                field = field_map.get(action)
                if field:
                    setattr(camera, field, True)

            self.add_camera_processor(camera)
        except Exception:
            logger.exception("[%s] Failed to boot processor from DB.", cam_name)

    # ------------------------------------------------------------------
    # Processor lifecycle
    # ------------------------------------------------------------------

    def _on_processor_stopped(self, cam_name: str) -> None:
        """
        Called by a processor when both pipelines are disabled.

        Runs on the MQTT callback thread, so joining the processor thread
        here is safe.
        """
        with self._lock:
            processor = self.camera_processors.pop(cam_name, None)
            if processor is not None:
                processor.stop()

            remaining: List[threading.Thread] = []
            target: Optional[threading.Thread] = None

            for t in self.threads:
                if t.name == f"Thread-{cam_name}":
                    target = t
                else:
                    remaining.append(t)
            self.threads = remaining

        if target is not None:
            target.join(timeout=5.0)
            if target.is_alive():
                logger.warning("[%s] Processor thread did not stop gracefully.", cam_name)

        logger.info("[%s] Processor stopped and cleaned up.", cam_name)

    async def _create_camera_processors(self):
        """
        Creates processor instances for each enabled camera defined in the database.
        """
        session_factory = get_sessionmaker()
        async with session_factory() as session:
            result = await session.execute(
                select(Camera).where(Camera.fight_enabled | Camera.vehicle_plate_enabled)
            )
            cameras = result.scalars().all()

        for camera in cameras:
            self.add_camera_processor(camera)

    def add_camera_processor(self, camera: Camera):
        """Create, register, and start a processor for *camera*."""
        with self._lock:
            if camera.name in self.camera_processors:
                logger.warning(f"Processor for camera '{camera.name}' already exists. Skipping.")
                return

            processor = CameraProcessor(
                cam_config=camera,
                detection_config=self.config.detection,
                system_config=self.config.system,
                predictor_wrapper=self.predictor_wrapper,
                mqtt_services=self.mqtt_services,
                thread_idx=len(self.threads),
                on_stop=self._on_processor_stopped,
            )
            self.camera_processors[camera.name] = processor

            # If manager is already running, start the processor thread
            if not self.stop_event.is_set():
                self._start_processor_thread(processor)

    def remove_camera_processor(self, camera_name: str):
        """Stop and deregister the processor for *camera_name*."""
        with self._lock:
            processor = self.camera_processors.pop(camera_name, None)
            if processor is None:
                return
            processor.stop()
            
            target_thread = None
            for i, thread in enumerate(self.threads):
                if thread.name == f"Thread-{camera_name}":
                    target_thread = self.threads.pop(i)
                    break
                    
        if target_thread:
            target_thread.join(timeout=2.0)
            
        logger.info("[%s] Processor stopped and removed.", camera_name)
        
    def update_camera_processor(self, camera: Camera):
        """Update an existing processor's settings in-place (no restart)."""
        processor = self.camera_processors.get(camera.name)

        if processor is None:
            logger.warning(f"No running processor for '{camera.name}' — creating new one.")
            self.add_camera_processor(camera)
            return

        processor.update_settings(camera)

    def _start_processor_thread(self, processor: CameraProcessor):
        thread = threading.Thread(
            target=processor.run, name=f"Thread-{processor.cam_name}"
        )
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
        logger.info(f"Camera processor '{processor.cam_name}' started")

    def _get_processor(self, name: str) -> Optional[CameraProcessor]:
        """
        Retrieves a camera processor by its name.

        Args:
            name (str): The name of the camera.

        Returns:
            Optional[CameraProcessor]: The processor instance, or None if not found.
        """
        return self.camera_processors.get(name)

    # ------------------------------------------------------------------
    # Manager lifecycle
    # ------------------------------------------------------------------

    async def start(self, blocking: bool = True):
        logger.info("Starting camera manager...")

        self.stop_event.clear()

        try:
            await self._create_camera_processors()

            logger.info(f"Created {len(self.camera_processors)} camera processors")

            logger.info("All camera threads started")
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Stopping...")
            self.stop()
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            self.stop()

            if not blocking:
                raise e

    def stop(self):
        """Stop all processors and cleanup"""
        self.stop_event.set()
        logger.info("Stopping system...")

        for processor in self.camera_processors.values():
            processor.stop()

        # Wait for all threads to finish with timeout
        try:
            for thread in self.threads:
                if thread.is_alive():
                    # Give each thread enough time to finish (5 seconds per camera)
                    thread.join(timeout=5)
                    if thread.is_alive():
                        logger.warning(
                            f"Warning: Thread {thread.name} did not stop gracefully"
                        )
        except KeyboardInterrupt:
            logger.warning("Force stopping (Ctrl+C during shutdown)...")
        finally:
            # Clear threads list to avoid re-joining
            self.threads.clear()
            self.camera_processors.clear()

        for service in self.mqtt_services.values():
            service.disconnect()

        logger.info("System stopped")
