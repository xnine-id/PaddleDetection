import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any


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
    config_path: str
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
class AppConfig:
    system: SystemConfig
    detection: DetectionConfig


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
            "config_path": str,
            "snapshot": {"enabled": bool, "output_dir": str},
            "mqtt": {
                "enabled": bool,
                "event_topic_prefix": str,
                "command_topic_prefix": str,
                "state_topic_prefix": str,
            },
        },
        "vehicle_plate": {
            "config_path": str,
            "snapshot": {"enabled": bool, "output_dir": str},
            "mqtt": {
                "enabled": bool,
                "event_topic_prefix": str,
                "command_topic_prefix": str,
                "state_topic_prefix": str,
            },
        },
    },
}

def _expand_project_root(obj: Any, project_root: str) -> Any:
    """Recursively replace '${PROJECT_ROOT}' placeholders in all string values."""
    if isinstance(obj, dict):
        return {k: _expand_project_root(v, project_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_project_root(v, project_root) for v in obj]
    if isinstance(obj, str):
        return obj.replace("${PROJECT_ROOT}", project_root)
    return obj


def _create_detection_module_config(
    module_dict: Dict[str, Any],
) -> DetectionModuleConfig:
    """Helper to create DetectionModuleConfig from dictionary"""
    return DetectionModuleConfig(
        config_path=module_dict["config_path"],
        snapshot=SnapshotConfig(**module_dict["snapshot"]),
        mqtt=MQTTConfig(**module_dict["mqtt"]),
    )


def load_config(config_file: str) -> AppConfig:
    """Load and validate configuration from YAML file"""
    # Accept either an absolute path or a path relative to the project root.
    # The project root is two directories above this file (src/utils/ -> project/).
    if not os.path.isabs(config_file):
        config_file = os.path.join(
            os.path.dirname(__file__), "..", "..", config_file
        )
    config_path = os.path.realpath(config_file)
    project_root = os.path.dirname(os.path.dirname(config_path))  # configs/ -> project/

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Expand ${PROJECT_ROOT} placeholders before validation
    config_dict = _expand_project_root(config_dict, project_root)
    validate_config(config_dict, REQUIRED_CONFIG)

    return AppConfig(
        system=SystemConfig(**config_dict["system"]),
        detection=DetectionConfig(
            fight=_create_detection_module_config(config_dict["detection"]["fight"]),
            vehicle_plate=_create_detection_module_config(
                config_dict["detection"]["vehicle_plate"]
            ),
        ),
    )
