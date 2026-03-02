import logging
import sys
import cv2
import uuid
from typing import Optional, Dict, Any
import threading
import os
from datetime import datetime

from internal.service.fight_tracker_int import FightTrackerInt
from internal.service.mqtt_service import MQTTService

logger = logging.getLogger("FIGHT_TRACKER")

class StreamFightTracker(FightTrackerInt):
    def __init__(self, snapshot_config: Dict[str, Any], cam_name: str, mqtt_service: Optional[MQTTService]):
        self.snapshot_enabled = snapshot_config['enabled']
        self.output_dir = snapshot_config['output_dir']
        self.cam_name = cam_name
        self.mqtt_service = mqtt_service

        self.current_score: Optional[int] = None
        self.threshold = 5
        self.update_count = 0
        self.event_id: Optional[str] = None

    def update(self, result: dict, frame):
        """Update current detections"""

        logger.debug(f"[{self.cam_name}] Result: {result}")

        if result and result["class"] == 1:
            self.update_count = 0
            prev_score = self.current_score
            self.current_score = result["score"]

            if prev_score == None or self.event_id == None:
                self.event_id = str(uuid.uuid4())

            if self.mqtt_service:
                snapshot = None
                if self.snapshot_enabled and frame is not None:
                    snapshot = self._save_snapshot(frame)

                self.mqtt_service.publish_event(
                    event_id=self.event_id,
                    cam_name=self.cam_name,
                    confidence=result["score"] * 100,
                    snapshot=snapshot,
                )

        else:
            self.update_count += 1
            if self.update_count % self.threshold == 0:
                prev_score = self.current_score
                self.current_score = None

                if prev_score != None:
                    self.event_id = str(uuid.uuid4())
                    self.mqtt_service.publish_event(
                        event_id=self.event_id,
                        cam_name=self.cam_name,
                        confidence=result["score"] * 100,
                        event_type="no_fight",
                    )

    def _save_snapshot(self, frame):
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        """Save snapshot in a separate thread to avoid blocking"""
        def save_task():
            try:
                final_output_dir = os.path.join(self.output_dir, today)
                
                os.makedirs(final_output_dir, exist_ok=True)
                
                final_output = os.path.join(final_output_dir, f"{self.cam_name}_{timestamp}.jpg")
                cv2.imwrite(final_output, frame)
            except Exception as e:
                logger.Info(f"[{self.cam_name}] Failed to save snapshot: {e}")

        # Run in background
        threading.Thread(target=save_task, daemon=True).start()
        
        # Return path immediately (predicted path)
        return f"/snapshots/{today}/{self.cam_name}_{timestamp}.jpg"
