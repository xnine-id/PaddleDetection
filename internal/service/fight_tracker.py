import logging
import uuid
from typing import Optional

from PaddleDetection.deploy.pipeline.datacollector import Result
from internal.service.mqtt_service import MQTTService

logger = logging.getLogger("FACE_TRACKER")

class FightTracker:
    def __init__(self, cam_name: str, mqtt_service: Optional[MQTTService]):
        self.cam_name = cam_name
        self.mqtt_service = mqtt_service
        self.current_score: Optional[int] = None
        self.threshold = 5
        self.update_count = 0

    def update(self, result: Result):
        """Update current detections"""

        action_result = result.get('video_action')

        if action_result and action_result["class"] == 1:
            self.update_count = 0
            prev_score = self.current_score
            self.current_score = action_result["score"]

            if self.mqtt_service and prev_score == None:
                self.mqtt_service.publish_event(
                    event_id=str(uuid.uuid4()),
                    cam_name=self.cam_name,
                    confidence=action_result["score"] * 100,
                    # snapshot=snapshot,
                )

        else:
            self.update_count += 1
            if self.update_count % self.threshold == 0:
                self.current_score = None

    def get_current_detections(self):
        """Get current detections for rendering"""
        return self.current_score
