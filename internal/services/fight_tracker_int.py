from abc import ABC, abstractmethod


class FightTrackerInt(ABC):
    @abstractmethod
    def update(self, result: dict, frame, frame_ids: list[int]):
        pass
