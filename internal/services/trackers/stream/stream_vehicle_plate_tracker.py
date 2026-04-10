from typing import Any
from collections import Counter
import logging
import uuid
import cv2
from typing import Dict, Optional, List, Set
import threading
import os
from datetime import datetime
import numpy as np

from internal.constants.infer_name import VEHICLE_PLATE
from internal.services.trackers.base.tracker_int import TrackerInt
from internal.services.mqtt.base.mqtt_service_int import MQTTServiceInt
from internal.utils.config_loader import SnapshotConfig

logger = logging.getLogger("VEHICLE_PLATE_TRACKER")

CLEANUP_INTERVAL = 30 # seconds
VEHICLE_ID_PRESENCE_TIMEOUT = 5 # seconds


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

        self.last_seen_vehicles: Dict[int, datetime] = {}
        self.last_cleanup = datetime.now()

        self.results: Dict[int, Dict[str, Any]] = {}

    def update(
        self, result: dict, frame: np.ndarray, frame_ids: Optional[List[int]] = None
    ):
        """
        Update the tracker with vehicle plate detection results.
        Updates heartbeat to prevent timeouts.
        """
        now = datetime.now()

        # Jalankan cleanup rutin setiap CLEANUP_INTERVAL detik untuk membersihkan state memori
        if (now - self.last_cleanup).total_seconds() > CLEANUP_INTERVAL:
            self._cleanup(now)
            self.last_cleanup = now

        # Get scores for each detected plates safely
        mot_res = result.get("mot", {}) or {}
        boxes = mot_res.get("boxes")
        plates_list: List[str] = (result.get("vehicleplate", {}) or {}).get("plate", [])

        should_save_snapshot = False
        snapshot = None
        mqtt_events_to_send = []

        for idx, box in enumerate(boxes):
            vehicle_id = int(box[0])
            score = float(box[2]) * 100

            # Update last seen for presence/timeout tracking
            self.last_seen_vehicles[vehicle_id] = now

            if idx < len(plates_list) and plates_list[idx] and plates_list[idx] != "":
                plate = plates_list[idx]

                event_type = None
                current_best_plate = None
                current_best_score = None

                if vehicle_id in self.results:
                    res = self.results[vehicle_id]
                    old_plates: List[str] = res["plates"]
                    old_scores: List[float] = res["scores"]
                    old_best_plate = res["best_plate"]

                    old_plates.append(plate)
                    old_scores.append(score)

                    counter = Counter(old_plates)
                    carlp = counter.most_common()
                    
                    current_best_plate = carlp[0][0] if carlp else plate
                    
                    # Find max score for the current best plate
                    max_scores = {}
                    for p, s in zip(old_plates, old_scores):
                        if p == current_best_plate and (p not in max_scores or s > max_scores[p]):
                            max_scores[p] = s
                    current_best_score = max_scores.get(current_best_plate, score)

                    self.results[vehicle_id] = {
                        "plates": old_plates,
                        "scores": old_scores,
                        "best_plate": current_best_plate,
                        "best_score": current_best_score,
                    }

                    if old_best_plate != current_best_plate:
                        event_type = "update"
                        should_save_snapshot = True
                else:
                    # New vehicle discovery
                    current_best_plate = plate
                    current_best_score = score
                    
                    self.results[vehicle_id] = {
                        "plates": [plate],
                        "scores": [score],
                        "best_plate": current_best_plate,
                        "best_score": current_best_score,
                    }
                    event_type = "enter"
                    should_save_snapshot = True

                if event_type:
                    mqtt_events_to_send.append({
                        "event_type": event_type,
                        "vehicle_id": vehicle_id,
                        "plate": current_best_plate,
                        "score": current_best_score
                    })

        if should_save_snapshot and self.snapshot_enabled:
            snapshot = self._save_snapshot(frame)

        if self.mqtt_service:
            for event in mqtt_events_to_send:
                event_id = str(uuid.uuid4())
                self.mqtt_service.publish_event(
                    event_id=event_id,
                    cam_name=self.cam_name,
                    confidence=event["score"],
                    snapshot=snapshot,
                    event_type=event["event_type"],
                    metadata={"plate": event["plate"], "vehicle_id": event["vehicle_id"]},
                )


    def reset(self):
        """Reset the vehicle plate tracker state."""
        self.last_seen_vehicles.clear()
        self.results.clear()

    def _cleanup(self, now: datetime):
        # Gunakan list comprehension agar tidak memodifikasi dictionary secara langsung ketika iterasi
        expired_vehicles = [
            vid for vid, last_time in self.last_seen_vehicles.items()
            if (now - last_time).total_seconds() > VEHICLE_ID_PRESENCE_TIMEOUT
        ]
        for vid in expired_vehicles:
            del self.last_seen_vehicles[vid]
            if vid in self.results:
                del self.results[vid]

    def _save_snapshot(self, frame: np.ndarray):
        """Save snapshot in a separate thread to avoid blocking"""
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        frame_copy = frame.copy()

        def save_task():
            try:
                final_output_dir = os.path.join(self.output_dir, today)

                os.makedirs(final_output_dir, exist_ok=True)

                final_output = os.path.join(
                    final_output_dir, f"{self.cam_name}_{timestamp}.jpg"
                )
                cv2.imwrite(final_output, frame_copy)
            except Exception as e:
                logger.exception(f"[{self.cam_name}] Failed to save snapshot: {e}")

        # Run in background
        threading.Thread(target=save_task, daemon=True).start()

        # Return path immediately (predicted path)
        return f"/snapshots/{VEHICLE_PLATE}/{today}/{self.cam_name}_{timestamp}.jpg"
