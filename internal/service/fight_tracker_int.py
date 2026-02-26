from abc import ABC, abstractmethod

class FightTrackerInt(ABC):
    @abstractmethod
    def update(self, result: dict, frame):
        pass