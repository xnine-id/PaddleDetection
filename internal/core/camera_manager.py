import threading
from typing import Dict, Any, List, Optional
import logging
from internal.core.fight_detector import FightDetector
from internal.core.camera_processor import CameraProcessor
from internal.service.mqtt_service import MQTTService

logger = logging.getLogger("CameraManager")


class CameraManager:
    def __init__(self, fight_detector: FightDetector, config: Dict[str, Any]):
        self.config = config
        self.fight_detector = fight_detector
        self.mqtt_service: Optional[MQTTService] = None

        if self.config.get("mqtt", {}).get("enabled", False):
            self.mqtt_service = MQTTService(self.config)

        self.threads: List[threading.Thread] = []
        self.camera_processors: Dict[str, CameraProcessor] = {}
        self.stop_event = threading.Event()

    def _create_camera_processors(self):
        pd_config = self.config.get("paddle_detection", {})
        snapshot_config = self.config.get("snapshot", {})

        for idx, cam in enumerate(self.config.get("cameras", [])):
            proc = CameraProcessor(
                cam_config=cam,
                snapshot_config=snapshot_config,
                pd_config=pd_config,
                fight_detector=self.fight_detector,
                mqtt_service=self.mqtt_service,
                thread_idx=idx,
                stop_event=self.stop_event,
            )
            self.camera_processors[cam["name"]] = proc

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
            logger.error(f"Unexpected error: {e}")
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

        if self.mqtt_service:
            self.mqtt_service.disconnect()

        logger.info("System stopped")
