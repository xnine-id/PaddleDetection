import json
import logging
from typing import Any, Dict
import numpy as np


logger = logging.getLogger("VEHICLE_PLATE_TRACKER")


class ImagesVehiclePlateTracker:
    def __init__(
        self,
    ):
        self.all_predictions: list[Dict[str, Any]] = []

    def update(self, result: dict, image_name: str):
        det_res = result.get("det", {})
        scores: np.ndarray = det_res["boxes"][:, 1]
        vehicleplate = result.get("vehicleplate", {}).get("vehicleplate", [])
        self.all_predictions.append(
            {
                "image_name": image_name,
                "plates": vehicleplate,
                "scores": scores.tolist(),
            }
        )

    def get_all_predictions(self):
        return self.all_predictions

    def save_all_predictions(self, json_output_path: str):
        """Save all predictions to a JSON file."""
        logger.info(
            f"save all predictions ({len(self.all_predictions)}) to {json_output_path}"
        )

        with open(json_output_path, "w") as f:
            json.dump(self.all_predictions, f, indent=2)
