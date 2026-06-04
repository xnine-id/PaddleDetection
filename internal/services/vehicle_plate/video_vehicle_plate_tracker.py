from typing import override
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional
from collections import Counter

from internal.services.base.tracker_int import TrackerInt

logger = logging.getLogger("API_VEHICLE_PLATE_TRACKER")


class VideoVehiclePlateTracker(TrackerInt):
    def __init__(self):
        self.all_predictions: list[Dict[str, Any]] = []
        self.vehicles: Dict[int, int] = {}

    @override
    def update(self, result: dict[str, Any], frame: np.ndarray[Any, Any], frame_ids: Optional[List[int]] = None):
        """Update current detections and store the prediction results."""
        mot_res = result.get("mot", {}) or {}
        boxes = mot_res.get("boxes", [])
        plates_list: List[str] = (result.get("vehicleplate", {}) or {}).get("plate", [])

        for idx, box in enumerate(boxes):
            if idx < len(plates_list) and plates_list[idx] and plates_list[idx] != "":
                vehicle_id: int = box[0]
                plate: str = plates_list[idx]
                score: float = box[2] * 100

                if vehicle_id in self.vehicles:
                    v_idx = self.vehicles[vehicle_id]
                    old_plates: List[str] = self.all_predictions[v_idx]["plates"]
                    old_scores: List[float] = self.all_predictions[v_idx]["scores"]
                    old_best_plates: List[str] = self.all_predictions[v_idx]["best_plates"]
                    old_best_scores: List[float] = self.all_predictions[v_idx]["best_scores"]

                    old_plates.append(plate)
                    old_scores.append(score)

                    counter = Counter(old_plates)
                    carlp = counter.most_common()
                    best_plate: Optional[str] = None

                    if len(carlp) > 0:
                        best_plate = carlp[0][0]

                    if best_plate and best_plate not in old_best_plates:
                        old_best_plates.append(best_plate)
                        max_scores: dict[str, float] = {}
                        for p, s in zip(old_plates, old_scores):
                            if p == best_plate and (p not in max_scores or s > max_scores[p]):
                                max_scores[p] = s
                        old_best_scores.append(max_scores[best_plate])

                    self.all_predictions[v_idx] = {
                        "vehicle_id": vehicle_id,
                        "plates": old_plates,
                        "scores": old_scores,
                        "best_plates": old_best_plates,
                        "best_scores": old_best_scores,
                    }
                else:
                    self.all_predictions.append({
                        "vehicle_id": vehicle_id,
                        "plates": [plate],
                        "scores": [score],
                        "best_plates": [plate],
                        "best_scores": [score],
                    })
                    self.vehicles[vehicle_id] = len(self.all_predictions) - 1

    @override
    def reset(self):
        """Reset the scores and predictions."""
        self.all_predictions = []
        self.vehicles = {}

    def get_all_predictions(self) -> list[Dict[str, Any]]:
        # transform plates and scores in all_predictions to just one plate and one score based on commonest for each vehicle id
        refined_predictions = []
        for pred in self.all_predictions:
            refined_predictions.append({
                "vehicle_id": pred['vehicle_id'],
                "plates": pred['best_plates'],
                "scores": pred['best_scores']
            })

        return refined_predictions

    def save_all_predictions(self, json_output_path: str):
        """Save all predictions to a JSON file."""

        logger.info(f"save all predictions ({len(self.all_predictions)}) to {json_output_path}")
        with open(json_output_path, "w") as f:
            json.dump(self.all_predictions, f, indent=2)    
