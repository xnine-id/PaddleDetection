import logging
import uuid
import cv2
from typing import Dict, Optional, List
import threading
import os
from datetime import datetime
import numpy as np

from internal.constants.infer_name import VEHICLE_PLATE
from internal.services.trackers.base.tracker_int import TrackerInt
from internal.services.mqtt.base.mqtt_service_int import MQTTServiceInt
from internal.utils.config_loader import SnapshotConfig

logger = logging.getLogger("VEHICLE_PLATE_TRACKER")


class StreamVehiclePlateTracker(TrackerInt):
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

        self.current_plates: set[str] = set()
        self.last_seen: Dict[str, datetime] = {}

    def update(
        self, result: dict, frame: np.ndarray, frame_ids: Optional[List[int]] = None
    ):
        """
        Update the tracker with vehicle plate detection results.
        Updates heartbeat to prevent timeouts.
        """
        now = datetime.now()

        # Get scores for each detected plates
        mot_res = result.get("mot", {}) or {}
        scores = mot_res["boxes"][:, 2]

        # Get list of plates from result and convert to set
        plates_list: List[str] = (result.get("vehicleplate", {}) or {}).get("plate", [])
        detected_plates = {d for d in plates_list if d}

        potential_new_entries = detected_plates - self.current_plates
        new_entries = []

        for plate in potential_new_entries:
            last_time = self.last_seen.get(plate)
            # If never seen or last seen > 1 second ago, it's a new entry
            if last_time is None or (now - last_time).total_seconds() > 1.0:
                new_entries.append(plate)

        snapshot = None

        if new_entries:
            if self.snapshot_enabled:
                snapshot = self._save_snapshot(frame)

        for plate in new_entries:
            if self.mqtt_service and snapshot and plate:
                event_id = str(uuid.uuid4())

                # Dapatkan score berdasarkan index di plates_list
                score = 0.0
                if plate in plates_list:
                    try:
                        idx = plates_list.index(plate)
                        if len(scores) > idx:
                            score = float(scores[idx]) * 100
                    except ValueError:
                        pass

                self.mqtt_service.publish_event(
                    event_id=event_id,
                    cam_name=self.cam_name,
                    confidence=score,
                    snapshot=snapshot,
                    event_type="enter",
                    metadata={"plate": plate},
                )

        # Update last seen for everyone currently detected
        for plate in detected_plates:
            self.last_seen[plate] = now

        self.current_plates = detected_plates

    def reset(self):
        """Reset the vehicle plate tracker state."""
        self.current_plates.clear()
        self.last_seen.clear()

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
        return f"/snapshots/{VEHICLE_PLATE}/{today}/{self.cam_name}_{timestamp}.jpg"
