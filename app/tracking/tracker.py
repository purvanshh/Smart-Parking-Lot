from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


DetectionBox = Tuple[float, float, float, float, float, int]  # x1,y1,x2,y2,conf,cls
TrackedObject = Tuple[float, float, float, float, int]  # x1,y1,x2,y2,track_id


@dataclass
class TrackerConfig:
    """
    Starter config placeholder.
    """


class Tracker:
    """
    Tracker wrapper.

    Starter behavior:
    - If you later integrate ByteTrack/supervision, implement `update()` to return tracked objects.
    """

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self._next_id = 1

    def update(self, detections: List[DetectionBox]) -> List[TrackedObject]:
        # Minimal placeholder: assigns new ID each detection each frame.
        tracked: List[TrackedObject] = []
        for x1, y1, x2, y2, _conf, _cls in detections:
            tracked.append((x1, y1, x2, y2, self._next_id))
            self._next_id += 1
        return tracked

