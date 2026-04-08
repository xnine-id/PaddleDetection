import logging
import cv2
import uuid
from typing import Optional, List
import threading
import os
from datetime import datetime
import numpy as np

from internal.constants.infer_name import VIDEO_ACTION
from internal.services.trackers.base.fight_tracker_int import FightTrackerInt
from internal.services.mqtt.base.mqtt_service_int import MQTTServiceInt
from internal.utils.config_loader import SnapshotConfig

logger = logging.getLogger("FIGHT_TRACKER")


class StreamFightTracker(FightTrackerInt):
    def __init__(
        self,
        snapshot_config: SnapshotConfig,
        cam_name: str,
        mqtt_service: Optional[MQTTServiceInt],
    ):
        self.snapshot_enabled = snapshot_config.enabled
        self.output_dir = snapshot_config.output_dir
        self.cam_name = cam_name
        self.mqtt_service = mqtt_service

        self.threshold = 5

        self.current_score: Optional[int] = None
        self.update_count = 0
        self.event_id: Optional[str] = None

    def update(self, result: dict, frame: np.ndarray, frame_ids: Optional[List[int]] = None):
        """
        Update current detections and check for fight events.
        Also updates the heartbeat timestamp to indicate the tracker is active.
        """
        if result and result["class"] == 1:
            self.update_count = 0
            prev_score = self.current_score
            self.current_score = result["score"]

            snapshot = None
            if prev_score == None or self.event_id == None:
                self.event_id = str(uuid.uuid4())
                if self.snapshot_enabled and frame is not None:
                    snapshot = self._save_snapshot(frame)

            if self.mqtt_service and self.current_score != prev_score:
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

    def reset(self):
        """Reset internal state of the tracker"""
        self.current_score = None
        self.event_id = None
        self.update_count = 0

    def _save_snapshot(self, frame: np.ndarray):
        """Save snapshot in a separate thread to avoid blocking"""
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        def save_task():
            try:
                final_output_dir = os.path.join(self.output_dir, today)

                os.makedirs(final_output_dir, exist_ok=True)

                final_output = os.path.join(
                    final_output_dir, f"{self.cam_name}_{timestamp}.jpg"
                )
                cv2.imwrite(final_output, frame)
            except Exception as e:
                logger.exception(f"[{self.cam_name}] Failed to save snapshot: {e}")

        # Run in background
        threading.Thread(target=save_task, daemon=True).start()

        # Return path immediately (predicted path)
        return f"/snapshots/{VIDEO_ACTION}/{today}/{self.cam_name}_{timestamp}.jpg"
