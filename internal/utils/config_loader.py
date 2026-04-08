import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class SnapshotConfig:
    enabled: bool
    output_dir: str


@dataclass
class MQTTConfig:
    enabled: bool
    event_topic_prefix: str
    command_topic_prefix: str
    state_topic_prefix: str


@dataclass
class DetectionModuleConfig:
    snapshot: SnapshotConfig
    mqtt: MQTTConfig


@dataclass
class DetectionConfig:
    fight: DetectionModuleConfig
    vehicle_plate: DetectionModuleConfig


@dataclass
class SystemConfig:
    config_path: str
    device: str
    output_dir: str
    pushurl_prefix: str


@dataclass
class CameraConfig:
    name: str
    url: str
    enabled: bool


@dataclass
class AppConfig:
    system: SystemConfig
    detection: DetectionConfig
    cameras: List[CameraConfig]


def validate_config(config: Dict[str, Any], required_keys: Dict[str, Any]) -> None:
    """Validate config structure recursively"""
    for key, expected in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

        if isinstance(expected, dict):
            if not isinstance(config[key], dict):
                raise ValueError(f"Config key '{key}' should be a dictionary")
            validate_config(config[key], expected)
        elif isinstance(expected, list):
            if not isinstance(config[key], list):
                raise ValueError(f"Config key '{key}' should be a list")
            if expected and isinstance(expected[0], dict):
                for item in config[key]:
                    validate_config(item, expected[0])


REQUIRED_CONFIG = {
    "system": {
        "config_path": str,
        "device": str,
        "output_dir": str,
        "pushurl_prefix": str,
    },
    "detection": {
        "fight": {
            "snapshot": {"enabled": bool, "output_dir": str},
            "mqtt": {
                "enabled": bool,
                "event_topic_prefix": str,
                "command_topic_prefix": str,
                "state_topic_prefix": str,
            },
        },
        "vehicle_plate": {
            "snapshot": {"enabled": bool, "output_dir": str},
            "mqtt": {
                "enabled": bool,
                "event_topic_prefix": str,
                "command_topic_prefix": str,
                "state_topic_prefix": str,
            },
        },
    },
    "cameras": [{"name": str, "url": str, "enabled": bool}],
}


def _create_detection_module_config(
    module_dict: Dict[str, Any],
) -> DetectionModuleConfig:
    """Helper to create DetectionModuleConfig from dictionary"""
    return DetectionModuleConfig(
        snapshot=SnapshotConfig(**module_dict["snapshot"]),
        mqtt=MQTTConfig(**module_dict["mqtt"]),
    )


def load_config(config_file: str) -> AppConfig:
    """Load and validate configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        validate_config(config, REQUIRED_CONFIG)

    return AppConfig(
        system=SystemConfig(**config["system"]),
        detection=DetectionConfig(
            fight=_create_detection_module_config(config["detection"]["fight"]),
            vehicle_plate=_create_detection_module_config(
                config["detection"]["vehicle_plate"]
            ),
        ),
        cameras=[CameraConfig(**cam) for cam in config["cameras"]],
    )
