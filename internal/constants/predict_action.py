from enum import Enum


class PredictAction(str, Enum):
    VIDEO_ACTION = "video_action"
    VEHICLE_PLATE = "vehicleplate"
