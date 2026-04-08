import threading
from typing import Dict, List
import logging
from internal.constants.infer_name import VEHICLE_PLATE, VIDEO_ACTION
from internal.core.predictor_wrapper import PredictorWrapper
from internal.core.camera_processor import CameraProcessor
from internal.services.mqtt.base.mqtt_service_int import MQTTServiceInt
from internal.services.mqtt.fight_mqtt_service import FightMQTTService
from internal.services.mqtt.plate_mqtt_service import PlateMQTTService
from internal.utils.config_loader import AppConfig

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

    def _create_camera_processors(self):
        for idx, cam_config in enumerate(self.config.cameras):
            proc = CameraProcessor(
                cam_config=cam_config,
                detection_config=self.config.detection,
                system_config=self.config.system,
                predictor_wrapper=self.predictor_wrapper,
                mqtt_services=self.mqtt_services,
                thread_idx=idx,
                stop_event=self.stop_event,
            )
            self.camera_processors[cam_config.name] = proc

    def start(self):
        logger.info("Starting camera manager...")

        try:
            self._create_camera_processors()
            logger.info(f"Created {len(self.camera_processors)} camera processors")

            for processor in self.camera_processors.values():
                logger.info(f"Starting thread for camera: {processor.cam_name}")
                thread = threading.Thread(target=processor.run, daemon=True)
                thread.start()
                self.threads.append(thread)

            logger.info("All camera threads started")
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Stopping...")
            self.stop()
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            self.stop()

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
