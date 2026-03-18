import json
import logging
from typing import Dict, Any

from internal.services.fight_tracker_int import FightTrackerInt

logger = logging.getLogger("API_FIGHT_TRACKER")


class VideoFightTracker(FightTrackerInt):
    def __init__(self):
        self.scores: list[int] = []
        self.all_predictions: list[Dict[str, Any]] = []

    def update(self, result: dict, frame, frame_ids: list[int]):
        """Update current detections"""

        self.all_predictions.append({
            "class": int(result["class"]),
            "score": float(result["score"]),
            "frame_ids": frame_ids.copy(),
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

    def save_all_predictions(self, json_output_path: str):
        with open(json_output_path, "w") as f:
            json.dump(self.all_predictions, f, indent=2)    
