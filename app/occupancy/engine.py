from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set, Tuple

from app.slots.slot_manager import SlotManager


TrackedObject = Tuple[float, float, float, float, int]  # x1,y1,x2,y2,track_id


@dataclass
class OccupancyEngine:
    slot_status: Dict[str, str] = field(default_factory=dict)  # slot_id -> "occupied"|"empty"
    vehicle_slots: Dict[int, str] = field(default_factory=dict)  # track_id -> slot_id
    entry_times: Dict[int, float] = field(default_factory=dict)  # track_id -> epoch seconds
    last_update_ts: Optional[float] = None

    def bootstrap_slots(self, slot_manager: SlotManager) -> None:
        for slot_id in slot_manager.all_slot_ids():
            self.slot_status.setdefault(slot_id, "empty")

    def update(self, tracked_objects: Iterable[TrackedObject], slot_manager: SlotManager) -> None:
        current_time = time.time()
        self.last_update_ts = current_time

        active_slots: Set[str] = set()
        seen_track_ids: Set[int] = set()

        for x1, y1, x2, y2, track_id in tracked_objects:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            slot = slot_manager.assign_slot(cx, cy)
            if not slot:
                continue

            active_slots.add(slot)
            seen_track_ids.add(track_id)

            if track_id not in self.entry_times:
                self.entry_times[track_id] = current_time
            self.vehicle_slots[track_id] = slot

        # Update slot statuses deterministically each tick
        for slot_id in slot_manager.all_slot_ids():
            self.slot_status[slot_id] = "occupied" if slot_id in active_slots else "empty"

        # Optional cleanup: drop old track IDs no longer present
        stale = [tid for tid in self.vehicle_slots.keys() if tid not in seen_track_ids]
        for tid in stale:
            self.vehicle_slots.pop(tid, None)
            self.entry_times.pop(tid, None)

    def summary(self) -> Dict[str, int]:
        total = len(self.slot_status)
        occupied = sum(1 for v in self.slot_status.values() if v == "occupied")
        return {"total_slots": total, "occupied": occupied, "available": total - occupied}

