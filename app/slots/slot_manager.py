from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.utils.geometry import point_in_polygon


Point = Tuple[float, float]
Polygon = List[Point]


@dataclass(frozen=True)
class SlotManager:
    slots: Dict[str, Polygon]  # slot_id -> polygon

    @staticmethod
    def from_json(path: str | Path) -> "SlotManager":
        data = json.loads(Path(path).read_text())
        slots = {slot_id: [(float(x), float(y)) for x, y in poly] for slot_id, poly in data["slots"].items()}
        return SlotManager(slots=slots)

    def assign_slot(self, cx: float, cy: float) -> Optional[str]:
        for slot_id, polygon in self.slots.items():
            if point_in_polygon(cx, cy, polygon):
                return slot_id
        return None

    def all_slot_ids(self) -> Iterable[str]:
        return self.slots.keys()

