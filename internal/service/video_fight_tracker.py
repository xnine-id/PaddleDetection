import logging
from typing import Dict, Any

from internal.service.fight_tracker_int import FightTrackerInt

logger = logging.getLogger("API_FIGHT_TRACKER")

class VideoFightTracker(FightTrackerInt):
    def __init__(self):
        self.scores: list[int] = []

    def update(self, result: dict, frame):
        """Update current detections"""

        logger.debug(f"Result: {result}")

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
