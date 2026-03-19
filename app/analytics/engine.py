from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.occupancy.engine import OccupancyEngine, ParkingSession


logger = logging.getLogger("smart_parking.analytics")


@dataclass(frozen=True)
class Alert:
    type: str
    timestamp: float
    metadata: Dict


@dataclass
class AnalyticsSnapshot:
    total_vehicles_served: int
    average_parking_duration_s: float
    average_duration_per_slot_s: Dict[str, float]
    slot_utilization_rate: Dict[str, float]  # slot_id -> utilization (0..1)
    busiest_slots: List[str]
    least_used_slots: List[str]


@dataclass
class AnalyticsEngine:
    """
    Incremental analytics engine.

    - Consumes OccupancyEngine.completed_sessions without reprocessing history
    - Maintains aggregates for fast API reads
    - Computes alerts from current occupancy state
    """

    occupancy_engine: OccupancyEngine
    busiest_k: int = 3
    overstay_threshold_s: float = 90.0
    almost_full_threshold: float = 0.10  # available/total < threshold

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _start_ts: Optional[float] = field(default=None, init=False)
    _last_processed_session_idx: int = field(default=0, init=False)

    # Aggregates
    _total_sessions: int = field(default=0, init=False)
    _duration_sum_s: float = field(default=0.0, init=False)
    _slot_duration_sum_s: Dict[str, float] = field(default_factory=dict, init=False)
    _slot_session_count: Dict[str, int] = field(default_factory=dict, init=False)

    # Cached outputs
    _snapshot: Optional[AnalyticsSnapshot] = field(default=None, init=False)
    _active_alerts: Dict[str, Alert] = field(default_factory=dict, init=False)  # key -> Alert

    def update(self, now_ts: Optional[float] = None) -> None:
        """
        Update aggregates and alerts.
        Call this periodically (e.g., every few seconds), not every frame.
        """
        now = time.time() if now_ts is None else now_ts
        with self._lock:
            if self._start_ts is None:
                self._start_ts = now

            self._ingest_completed_sessions()
            self._recompute_snapshot(now)
            self._recompute_alerts(now)

    def snapshot(self) -> AnalyticsSnapshot:
        with self._lock:
            if self._snapshot is None:
                self._snapshot = AnalyticsSnapshot(
                    total_vehicles_served=0,
                    average_parking_duration_s=0.0,
                    average_duration_per_slot_s={},
                    slot_utilization_rate={},
                    busiest_slots=[],
                    least_used_slots=[],
                )
            return self._snapshot

    def alerts(self) -> List[Alert]:
        with self._lock:
            return list(self._active_alerts.values())

    def recent_sessions(self, limit: int = 50) -> List[ParkingSession]:
        # completed_sessions is append-only; copying a tail is cheap.
        sessions = self.occupancy_engine.completed_sessions
        return sessions[-limit:] if limit > 0 else []

    def _ingest_completed_sessions(self) -> None:
        sessions = self.occupancy_engine.completed_sessions
        n = len(sessions)
        if self._last_processed_session_idx >= n:
            return

        new_sessions = sessions[self._last_processed_session_idx :]
        self._last_processed_session_idx = n

        for s in new_sessions:
            self._total_sessions += 1
            self._duration_sum_s += s.duration_s
            self._slot_duration_sum_s[s.slot_id] = self._slot_duration_sum_s.get(s.slot_id, 0.0) + s.duration_s
            self._slot_session_count[s.slot_id] = self._slot_session_count.get(s.slot_id, 0) + 1

        logger.info("analytics_ingest new_sessions=%s total_sessions=%s", len(new_sessions), self._total_sessions)

    def _recompute_snapshot(self, now: float) -> None:
        # Average duration overall
        avg = (self._duration_sum_s / self._total_sessions) if self._total_sessions else 0.0

        # Average duration per slot
        avg_per_slot: Dict[str, float] = {}
        for slot_id, sum_s in self._slot_duration_sum_s.items():
            cnt = self._slot_session_count.get(slot_id, 0)
            avg_per_slot[slot_id] = (sum_s / cnt) if cnt else 0.0

        # Utilization per slot: (occupied time) / (total time since start)
        start = self._start_ts or now
        total_window_s = max(1e-6, now - start)

        util: Dict[str, float] = {}
        # Base utilization from completed sessions.
        for slot_id, sum_s in self._slot_duration_sum_s.items():
            util[slot_id] = min(1.0, max(0.0, sum_s / total_window_s))

        # Add ongoing (active) occupancy time (best effort).
        # If a slot is currently occupied, add (now - entry_ts) to its occupied time.
        for slot_id, track_id in self.occupancy_engine.slot_track.items():
            if track_id is None:
                continue
            entry_ts = self.occupancy_engine.entry_times.get(track_id)
            if entry_ts is None:
                continue
            active_s = max(0.0, now - entry_ts)
            base_s = self._slot_duration_sum_s.get(slot_id, 0.0)
            util[slot_id] = min(1.0, max(0.0, (base_s + active_s) / total_window_s))

        # Busiest / least-used slots by total occupied time (completed + active)
        slot_scores = sorted(util.items(), key=lambda kv: kv[1], reverse=True)
        busiest = [slot_id for slot_id, _ in slot_scores[: self.busiest_k]]
        least = [slot_id for slot_id, _ in slot_scores[-self.busiest_k :]] if slot_scores else []

        self._snapshot = AnalyticsSnapshot(
            total_vehicles_served=self._total_sessions,
            average_parking_duration_s=avg,
            average_duration_per_slot_s=avg_per_slot,
            slot_utilization_rate=util,
            busiest_slots=busiest,
            least_used_slots=least,
        )

    def _recompute_alerts(self, now: float) -> None:
        alerts: Dict[str, Alert] = {}

        # lot_almost_full
        summary = self.occupancy_engine.summary()
        total = summary["total_slots"]
        available = summary["available"]
        if total > 0 and (available / total) < self.almost_full_threshold:
            key = "lot_almost_full"
            alerts[key] = Alert(
                type="lot_almost_full",
                timestamp=now,
                metadata={"available": available, "total": total, "ratio": available / total},
            )

        # overstay (active vehicles exceeding threshold)
        for track_id, duration_s in self.occupancy_engine.durations_s.items():
            # Only consider tracks that are still active (recently seen)
            last_seen = self.occupancy_engine.last_seen_ts.get(track_id)
            if last_seen is None or (now - last_seen) > self.occupancy_engine.clear_timeout_s:
                continue
            if duration_s >= self.overstay_threshold_s:
                slot_id = self.occupancy_engine.vehicle_slots.get(track_id)
                key = f"overstay:{track_id}"
                alerts[key] = Alert(
                    type="overstay",
                    timestamp=now,
                    metadata={"track_id": track_id, "slot_id": slot_id, "duration_s": duration_s},
                )

        # Log newly triggered alerts (best effort)
        for k, a in alerts.items():
            if k not in self._active_alerts:
                logger.warning("alert_trigger type=%s metadata=%s", a.type, a.metadata)

        self._active_alerts = alerts

