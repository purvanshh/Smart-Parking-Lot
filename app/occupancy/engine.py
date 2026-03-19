from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from app.slots.slot_manager import SlotManager


TrackedObject = Tuple[float, float, float, float, int]  # x1,y1,x2,y2,track_id


logger = logging.getLogger("smart_parking.occupancy")


@dataclass(frozen=True)
class ParkingSession:
    track_id: int
    slot_id: str
    entry_ts: float
    exit_ts: float
    duration_s: float


@dataclass
class OccupancyEngine:
    slot_status: Dict[str, str] = field(default_factory=dict)  # slot_id -> "occupied"|"empty"
    # Stable assignment state (derived from tracking, not raw detections)
    slot_track: Dict[str, Optional[int]] = field(default_factory=dict)  # slot_id -> current track_id (or None)
    vehicle_slots: Dict[int, str] = field(default_factory=dict)  # track_id -> current slot_id

    # Time bookkeeping (epoch seconds)
    entry_times: Dict[int, float] = field(default_factory=dict)  # track_id -> entry timestamp
    last_seen_ts: Dict[int, float] = field(default_factory=dict)  # track_id -> last time seen by tracker
    durations_s: Dict[int, float] = field(default_factory=dict)  # track_id -> duration (active or final)
    completed_sessions: List[ParkingSession] = field(default_factory=list)

    # Temporal smoothing / debouncing
    debounce_frames: int = 4
    clear_timeout_s: float = 1.5
    _pending_state: Dict[str, str] = field(default_factory=dict)  # slot_id -> "occupied"|"empty"
    _pending_count: Dict[str, int] = field(default_factory=dict)  # slot_id -> consecutive frames observed
    last_update_ts: Optional[float] = None

    def bootstrap_slots(self, slot_manager: SlotManager) -> None:
        for slot_id in slot_manager.all_slot_ids():
            self.slot_status.setdefault(slot_id, "empty")
            self.slot_track.setdefault(slot_id, None)
            self._pending_state.setdefault(slot_id, "empty")
            self._pending_count.setdefault(slot_id, 0)

    def update(
        self,
        tracked_objects: Iterable[TrackedObject],
        slot_manager: SlotManager,
        now_ts: Optional[float] = None,
    ) -> None:
        """
        Update occupancy based on tracked objects.

        Design goals:
        - Stable per-slot occupancy (debounced; avoids flicker)
        - Stable track_id → slot assignment (noise tolerant)
        - Entry/exit timestamps and durations
        - Occlusion tolerance (timeout before clearing)
        """
        now = time.time() if now_ts is None else now_ts
        self.last_update_ts = now

        # Track IDs observed this update
        seen_track_ids: Set[int] = set()

        # Slot observed as occupied by track_id this frame (best effort).
        observed_slot_to_track: Dict[str, int] = {}

        for x1, y1, x2, y2, track_id in tracked_objects:
            seen_track_ids.add(track_id)
            self.last_seen_ts[track_id] = now

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            slot = slot_manager.assign_slot(cx, cy)

            # If mapping fails due to minor noise, keep last known slot briefly.
            if slot is None and track_id in self.vehicle_slots:
                slot = self.vehicle_slots[track_id]

            if slot is None:
                continue

            # If multiple tracks map to same slot, keep first seen.
            observed_slot_to_track.setdefault(slot, track_id)

            # First assignment to any slot = entry event for this track_id.
            if track_id not in self.entry_times:
                self.entry_times[track_id] = now
                logger.info("entry track_id=%s slot=%s", track_id, slot)

            self.vehicle_slots[track_id] = slot

        # Observed occupancy state with occlusion tolerance.
        observed_state: Dict[str, str] = {}
        for slot_id in slot_manager.all_slot_ids():
            if slot_id in observed_slot_to_track:
                observed_state[slot_id] = "occupied"
                continue

            prev_tid = self.slot_track.get(slot_id)
            if prev_tid is not None:
                last_seen = self.last_seen_ts.get(prev_tid)
                if last_seen is not None and (now - last_seen) <= self.clear_timeout_s:
                    observed_state[slot_id] = "occupied"
                    continue

            observed_state[slot_id] = "empty"

        # Debounce state changes (N consecutive observations before commit).
        for slot_id in slot_manager.all_slot_ids():
            desired = observed_state[slot_id]
            if self._pending_state.get(slot_id) != desired:
                self._pending_state[slot_id] = desired
                self._pending_count[slot_id] = 1
            else:
                self._pending_count[slot_id] = self._pending_count.get(slot_id, 0) + 1

            if self._pending_count[slot_id] >= self.debounce_frames:
                self.slot_status[slot_id] = desired

        # Commit stable slot->track assignment + finalize exits on stable empty.
        for slot_id in slot_manager.all_slot_ids():
            if self.slot_status.get(slot_id) == "occupied":
                if slot_id in observed_slot_to_track:
                    new_tid = observed_slot_to_track[slot_id]
                    old_tid = self.slot_track.get(slot_id)
                    if old_tid != new_tid:
                        logger.info("slot_assign slot=%s track_id=%s prev=%s", slot_id, new_tid, old_tid)
                    self.slot_track[slot_id] = new_tid
                # else: keep previous assignment during occlusion timeout window
            else:
                old_tid = self.slot_track.get(slot_id)
                if old_tid is not None:
                    self._finalize_track(old_tid, exit_ts=now, reason=f"slot_empty:{slot_id}")
                self.slot_track[slot_id] = None

        # Finalize tracks that disappeared (timeout), even if their slot never debounced empty.
        for tid in list(self.vehicle_slots.keys()):
            last_seen = self.last_seen_ts.get(tid)
            if last_seen is None:
                continue
            if (now - last_seen) > self.clear_timeout_s:
                self._finalize_track(tid, exit_ts=now, reason="track_timeout")

        # Update active durations for tracks still considered present.
        for tid, entry_ts in list(self.entry_times.items()):
            last_seen = self.last_seen_ts.get(tid)
            if last_seen is None:
                continue
            if (now - last_seen) <= self.clear_timeout_s:
                self.durations_s[tid] = max(0.0, now - entry_ts)

    def summary(self) -> Dict[str, int]:
        total = len(self.slot_status)
        occupied = sum(1 for v in self.slot_status.values() if v == "occupied")
        return {"total_slots": total, "occupied": occupied, "available": total - occupied}

    def _finalize_track(self, track_id: int, exit_ts: float, reason: str) -> None:
        """
        Close out a track's parking session if we have enough information.
        """
        slot = self.vehicle_slots.pop(track_id, None)
        entry_ts = self.entry_times.pop(track_id, None)
        self.last_seen_ts.pop(track_id, None)

        if entry_ts is None or slot is None:
            return

        duration = max(0.0, exit_ts - entry_ts)
        self.durations_s[track_id] = duration
        self.completed_sessions.append(
            ParkingSession(
                track_id=track_id,
                slot_id=slot,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                duration_s=duration,
            )
        )
        logger.info("exit track_id=%s slot=%s duration_s=%.2f reason=%s", track_id, slot, duration, reason)

