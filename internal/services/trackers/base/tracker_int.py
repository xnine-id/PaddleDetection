import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List


class TrackerInt(ABC):
    """
    Abstract base class for all tracker implementations.
    Defines the standard interface for updating tracker states, resetting,
    and handling heartbeats for health checks.
    """
    @abstractmethod
    def update(self, result: dict, frame: np.ndarray, frame_ids: Optional[List[int]] = None):
        """Update tracker with the latest detection result."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the internal state of the tracker."""
        pass
