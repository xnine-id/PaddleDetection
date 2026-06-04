from typing import Optional
import threading
from typing import Dict, List
import logging
from internal.constants.infer_name import VEHICLE_PLATE, VIDEO_ACTION
from internal.core.predictor_wrapper import PredictorWrapper
from internal.core.camera_processor import CameraProcessor
from internal.services.base.mqtt_service_int import MQTTServiceInt
from internal.services.fight.fight_mqtt_service import FightMQTTService
from internal.services.vehicle_plate.plate_mqtt_service import PlateMQTTService
from internal.utils.config_loader import AppConfig
from internal.database.entity.camera import Camera

logger = logging.getLogger("CameraManager")


class CameraManager:
    """
    Manages multiple camera processors, initializes them based on the configuration,
    and handles their threading lifecycle (start/stop).
    """
    def __init__(self, predictor_wrapper: PredictorWrapper, config: AppConfig):
        self.config = config
        self.predictor_wrapper = predictor_wrapper
        self.mqtt_services: Dict[str, MQTTServiceInt] = {}

        # Initialize MQTT services for each detection module if enabled
        if self.config.detection.fight.mqtt.enabled:
            self.mqtt_services[VIDEO_ACTION] = FightMQTTService(self.config.detection.fight.mqtt)

        if self.config.detection.vehicle_plate.mqtt.enabled:
            self.mqtt_services[VEHICLE_PLATE] = PlateMQTTService(self.config.detection.vehicle_plate.mqtt)

        self.threads: List[threading.Thread] = []
        self.camera_processors: Dict[str, CameraProcessor] = {}
        self.stop_event = threading.Event()

    async def _create_camera_processors(self):
        """
        Creates processor instances for each enabled camera defined in the database.
        """
        from internal.database.session import get_sessionmaker
        from sqlalchemy.future import select

        session_factory = get_sessionmaker()
        async with session_factory() as session:
            result = await session.execute(select(Camera))
            cameras = result.scalars().all()

            for camera in cameras:
                self.add_camera_processor(camera)

    def add_camera_processor(self, camera: Camera):
        """Adds and starts a new camera processor."""
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
        )
        self.camera_processors[camera.name] = processor
        
        # If manager is already running, start the processor thread
        if not self.stop_event.is_set():
            self._start_processor_thread(processor)

    def update_camera_processor(self, camera: Camera):
        """Updates an existing camera processor by restarting it with new config."""
        self.remove_camera_processor(camera.name)
        self.add_camera_processor(camera)

    def remove_camera_processor(self, camera_name: str):
        """Stops and removes a camera processor."""
        processor = self.camera_processors.pop(camera_name, None)
        if processor:
            processor.stop()
            # Find and join thread (optional, but good for cleanup)
            for i, thread in enumerate(self.threads):
                if thread.name == f"Thread-{camera_name}":
                    if thread.is_alive():
                        thread.join(timeout=2.0)
                    self.threads.pop(i)
                    break
            logger.info(f"Camera processor '{camera_name}' stopped and removed")

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
