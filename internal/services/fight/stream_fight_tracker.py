import logging
import cv2
import uuid
from typing import Optional, List, Any, override
import threading
import os
from datetime import datetime
import numpy as np

from internal.constants.predict_action import PredictAction
from internal.services.base.tracker_int import TrackerInt
from internal.services.base.mqtt_service_int import MQTTServiceInt
from internal.utils.config_loader import SnapshotConfig

logger = logging.getLogger("FIGHT_TRACKER")

FIGHT_TIME_THRESHOLD = 10  # seconds


class StreamFightTracker(TrackerInt):
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

        self.event_id: Optional[str] = None
        self.last_fight_time: Optional[datetime] = None

    @override
    def update(
        self, result: dict[str, Any], frame: np.ndarray[Any, Any], frame_ids: List[int]
    ):
        """
        Update current detections and check for fight events.
        Also updates the heartbeat timestamp to indicate the tracker is active.
        """
        now = datetime.now()
        is_new_event = (
            self.last_fight_time is None
            or self.event_id is None
            or (now - self.last_fight_time).seconds > FIGHT_TIME_THRESHOLD
        )

        if result:
            is_fight = result["class"] == 1
            confidence = result["score"] * 100
            event_type = None
            should_save_snapshot = False

            if is_fight:
                self.last_fight_time = now
                event_type = "fight"
                if is_new_event:
                    self.event_id = str(uuid.uuid4())
                    should_save_snapshot = True
            else:
                if is_new_event:
                    self.event_id = str(uuid.uuid4())
                    event_type = "no_fight"

            event_id_str = self.event_id or str(uuid.uuid4())
            event_type_str = event_type or "no_fight"

            def mqtt_task(frame: np.ndarray[Any, Any]):
                snapshot = None
                if should_save_snapshot and self.snapshot_enabled and frame is not None:
                    snapshot = self._save_snapshot(frame)

                if self.mqtt_service:
                    self.mqtt_service.publish_event(
                        event_id=event_id_str,
                        cam_name=self.cam_name,
                        confidence=confidence,
                        snapshot=snapshot,
                        event_type=event_type_str,
                    )

            # Send mqtt event every new event and fight event (for update score)
            if is_new_event or is_fight:
                threading.Thread(target=mqtt_task, args=(frame.copy(),), daemon=True).start()

    @override
    def reset(self):
        """Reset internal state of the tracker"""
        self.event_id = None
        self.last_fight_time = None

    def _save_snapshot(self, frame: np.ndarray[Any, Any]):
        """Save snapshot in a separate thread to avoid blocking"""
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:19] # include ms

        try:
            final_output_dir = os.path.join(self.output_dir, today)

            os.makedirs(final_output_dir, exist_ok=True)

            final_output = os.path.join(
                final_output_dir, f"{self.cam_name}_{timestamp}.jpg"
            )
            cv2.imwrite(final_output, frame)  # type: ignore
        except Exception as e:
            logger.exception(f"[{self.cam_name}] Failed to save snapshot: {e}")
            return None

        # Return path immediately (predicted path)
        return f"/snapshots/{PredictAction.VIDEO_ACTION}/{today}/{self.cam_name}_{timestamp}.jpg"
