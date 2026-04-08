import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from internal.utils.config_loader import MQTTConfig
from internal.services.mqtt.base.mqtt_service_int import MQTTServiceInt

logger = logging.getLogger("PlateMQTT")


class PlateMQTTService(MQTTServiceInt):
    """MQTT Service for publishing vehicle plate detection events"""

    def __init__(self, config: MQTTConfig):
        super().__init__(config, service_name="PlateMQTT")

    def publish_event(
        self,
        event_id: str,
        cam_name: str,
        confidence: float,
        snapshot: Optional[str] = None,
        event_type: str = "enter",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.client:
            return

        payload = {
            "event_id": event_id,
            "camera": cam_name,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "snapshot": snapshot,
        }

        if metadata:
            payload["metadata"] = metadata

        topic = f"{self.event_topic}/{cam_name}"
        self.client.publish(topic, json.dumps(payload), qos=1)
