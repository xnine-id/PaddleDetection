import json
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from paho.mqtt import client as mqtt

from internal.utils.config_loader import MQTTConfig

logger = logging.getLogger("MQTTService")


class MQTTServiceInt(ABC):
    def __init__(self, config: MQTTConfig, service_name: str = "MQTT"):
        self.config = config
        self.service_name = service_name
        self.enabled = config.enabled
        self.event_topic = config.event_topic_prefix
        self.cmd_prefix = config.command_topic_prefix
        self.state_prefix = config.state_topic_prefix

        self.client: Optional[mqtt.Client] = None
        self.command_callbacks: Dict[str, Any] = {}

        if self.enabled:
            self._setup_client()

    def _setup_client(self):
        """Initialize and connect MQTT client"""
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        username = os.getenv("MQTT_AUTH_USERNAME")
        password = os.getenv("MQTT_AUTH_PASSWORD")
        host = os.getenv("MQTT_HOST", "localhost")
        port = int(os.getenv("MQTT_PORT", 1883))

        if username and password:
            self.client.username_pw_set(username, password)

        try:
            self.client.connect(host, port)
            self.client.loop_start()
            logger.info(f"[{self.service_name}] Connected to {host}:{port}")
        except Exception as e:
            logger.error(f"[{self.service_name}] Connection failed: {e}")
            self.client = None

    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection and resubscribe to topics"""
        logger.debug(f"[{self.service_name}] Connected with result code {rc}")
        if self.client:
            self.client.subscribe(f"{self.cmd_prefix}/#")
            logger.info(f"[{self.service_name}] Subscribed to {self.cmd_prefix}/#")

    def _on_message(self, client, userdata, msg):
        """Central message dispatcher"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            if topic.startswith(self.cmd_prefix):
                cam_name = topic.replace(f"{self.cmd_prefix}/", "")
                if cam_name in self.command_callbacks:
                    self.command_callbacks[cam_name](topic, payload)
        except Exception as e:
            logger.error(f"[{self.service_name}] Error handling message on {msg.topic}: {e}")

    def register_camera(self, cam_name: str, state: bool, on_command_callback: Any):
        """Register a camera for commands and initial state"""
        self.command_callbacks[cam_name] = on_command_callback
        if self.client:
            self.publish_state(cam_name, state)

    @abstractmethod
    def publish_event(
        self,
        event_id: str,
        cam_name: str,
        confidence: float,
        snapshot: Optional[str] = None,
        event_type: str = "fight",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Publish detection event"""
        pass

    def publish_state(self, cam_name: str, is_running: bool):
        """Publish camera running state"""
        if not self.client:
            return
        topic = f"{self.state_prefix}/{cam_name}"
        self.client.publish(topic, json.dumps(is_running), retain=True)
        logger.debug(f"[{self.service_name}] State: {cam_name} is {'RUNNING' if is_running else 'STOPPED'}")

    def disconnect(self):
        """Cleanup connection"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info(f"[{self.service_name}] Disconnected")
