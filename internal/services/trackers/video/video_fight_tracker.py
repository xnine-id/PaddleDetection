import json
import logging
from typing import Dict, Any, List, Optional

from internal.services.trackers.base.fight_tracker_int import FightTrackerInt

logger = logging.getLogger("API_FIGHT_TRACKER")


class VideoFightTracker(FightTrackerInt):
    def __init__(self):
        self.scores: list[int] = []
        self.all_predictions: list[Dict[str, Any]] = []

    def update(self, result: dict, frame, frame_ids: Optional[List[int]] = None):
        """Update current detections and store the prediction results."""
        ids_copy = frame_ids.copy() if frame_ids is not None else []

        self.all_predictions.append({
            "class": int(result["class"]),
            "score": float(result["score"]),
            "frame_ids": ids_copy,
        })

        if result and result["class"] == 1:
            self.scores.append(result["score"])

    def get_scores(self) -> list[int]:
        return self.scores

    def get_avg_scores(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0

    def get_highest_score(self) -> float:
        return max(self.scores) if self.scores else 0

    def get_lowest_score(self) -> float:
        return min(self.scores) if self.scores else 0

    def reset(self):
        """Reset the scores and predictions."""
        self.scores = []
        self.all_predictions = []

    def save_all_predictions(self, json_output_path: str):
        """Save all predictions to a JSON file."""

        logger.info(f"save all predictions ({len(self.all_predictions)}) to {json_output_path}")
        with open(json_output_path, "w") as f:
            json.dump(self.all_predictions, f, indent=2)    
