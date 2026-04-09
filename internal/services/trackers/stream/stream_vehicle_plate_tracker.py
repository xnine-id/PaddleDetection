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

VEHICLE_ID_PRESENCE_TIMEOUT = 10 # seconds
PLATE_PRESENCE_TIMEOUT = 5 # seconds
CLEANUP_INTERVAL = 30 # seconds


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
        self.last_seen: Dict[str, datetime] = {}
        self._last_cleanup = datetime.now()

    def update(
        self, result: dict, frame: np.ndarray, frame_ids: Optional[List[int]] = None
    ):
        """
        Update the tracker with vehicle plate detection results.
        Updates heartbeat to prevent timeouts.
        """
        now = datetime.now()

        # Jalankan cleanup rutin setiap CLEANUP_INTERVAL detik untuk membersihkan state memori
        if (now - self._last_cleanup).total_seconds() > CLEANUP_INTERVAL:
            self._cleanup(now)
            self._last_cleanup = now

        # Get scores for each detected plates safely
        mot_res = result.get("mot", {}) or {}
        boxes = mot_res.get("boxes")
        if boxes is not None and len(boxes) > 0:
            ids = boxes[:, 0]
            scores = boxes[:, 2]
        else:
            ids = []
            scores = []

        # Get list of plates from result and convert to set
        plates_list: List[str] = (result.get("vehicleplate", {}) or {}).get("plate", [])
        detected_plates = {d for d in plates_list if d}

        new_entries = []

        for plate in detected_plates:
            last_time = self.last_seen.get(plate)
            # If never seen or last seen > PLATE_PRESENCE_TIMEOUT second ago, it's a new entry
            if last_time is None or (now - last_time).total_seconds() > PLATE_PRESENCE_TIMEOUT:
                new_entries.append(plate)

        snapshot = None

        if new_entries and self.snapshot_enabled:
            snapshot = self._save_snapshot(frame)

        for plate in new_entries:
            if self.mqtt_service and plate:
                event_id = str(uuid.uuid4())

                # Dapatkan score berdasarkan index di plates_list
                score = 0.0
                vehicle_id = None
                if plate in plates_list:
                    try:
                        idx = plates_list.index(plate)
                        if len(scores) > idx:
                            score = float(scores[idx]) * 100
                            vehicle_id = int(ids[idx])
                    except ValueError:
                        pass

                event_type = "enter"

                if vehicle_id is not None:
                    last_time = self.last_seen_vehicles.get(vehicle_id)
                    # Jika kendaraan sudah terlihat sebelumnya dalam waktu dekat, maka ini merupakan update
                    if last_time is not None and (now - last_time).total_seconds() <= VEHICLE_ID_PRESENCE_TIMEOUT:
                        event_type = "update"

                    self.last_seen_vehicles[vehicle_id] = now

                self.mqtt_service.publish_event(
                    event_id=event_id,
                    cam_name=self.cam_name,
                    confidence=score,
                    snapshot=snapshot,
                    event_type=event_type,
                    metadata={"plate": plate, "vehicle_id": vehicle_id},
                )

        # Update last seen for everyone currently detected
        for plate in detected_plates:
            self.last_seen[plate] = now


    def reset(self):
        """Reset the vehicle plate tracker state."""
        self.last_seen.clear()
        self.last_seen_vehicles.clear()

    def _cleanup(self, now: datetime):
        # Gunakan list comprehension agar tidak memodifikasi dictionary secara langsung ketika iterasi
        expired_vehicles = [
            vid for vid, last_time in self.last_seen_vehicles.items()
            if (now - last_time).total_seconds() > VEHICLE_ID_PRESENCE_TIMEOUT
        ]
        for vid in expired_vehicles:
            del self.last_seen_vehicles[vid]
            
        # Cleanup terhadap history plat nomor yang sudah lama tidak terlihat (misal > 60 detik)
        expired_plates = [
            plate for plate, last_time in self.last_seen.items()
            if (now - last_time).total_seconds() > 60
        ]
        for plate in expired_plates:
            del self.last_seen[plate]

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
