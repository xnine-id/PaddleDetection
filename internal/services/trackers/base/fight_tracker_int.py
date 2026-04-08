from abc import abstractmethod
from typing import Optional, List
from internal.services.trackers.base.tracker_int import TrackerInt


class FightTrackerInt(TrackerInt):
    """
    Interface specifically designed for fight/action tracking implementations.
    Inherits from the base TrackerInt.
    """
    @abstractmethod
    def update(self, result: dict, frame, frame_ids: Optional[List[int]] = None):
        """Update fight detection results."""
        pass
