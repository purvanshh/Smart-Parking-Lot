from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


DetectionBox = Tuple[float, float, float, float, float, int]  # x1,y1,x2,y2,conf,cls
TrackedObject = Tuple[float, float, float, float, int]  # x1,y1,x2,y2,track_id


logger = logging.getLogger("smart_parking.tracker")


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    last_seen_ts: float
    hits: int = 1
    # When a track isn't matched in a given update, we keep it around until timeout.
    misses: int = 0


@dataclass
class TrackerConfig:
    """
    IoU-based tracker config (ByteTrack-ready interface).

    This is intentionally simple and fast, but provides:
    - consistent `track_id` across frames
    - tolerance to brief occlusion (via `max_age_seconds`)
    """
    iou_threshold: float = 0.3
    max_age_seconds: float = 1.5
    min_hits_to_confirm: int = 1
    log_assignments: bool = False


class Tracker:
    """
    Tracker wrapper.

    Level 2 tracker:
    - Greedy IoU association
    - Stable IDs across frames
    - Output contract remains compatible with ByteTrack integration:
      (x1, y1, x2, y2, track_id)
    """

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def update(self, detections: List[DetectionBox], now_ts: Optional[float] = None) -> List[TrackedObject]:
        """
        Args:
            detections: list of (x1,y1,x2,y2,conf,cls)
            now_ts: optional timestamp to keep time consistent across pipeline stages
        """
        now = time.time() if now_ts is None else now_ts

        det_bboxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _conf, _cls in detections]
        det_used = [False] * len(det_bboxes)

        # Build all candidate matches (track, det, iou), then greedy pick highest IoU.
        candidates: List[Tuple[float, int, int]] = []
        track_ids = list(self._tracks.keys())
        for tid in track_ids:
            tb = self._tracks[tid].bbox
            for di, db in enumerate(det_bboxes):
                iou = _iou(tb, db)
                if iou >= self.config.iou_threshold:
                    candidates.append((iou, tid, di))
        candidates.sort(reverse=True, key=lambda x: x[0])

        matched_tracks: Dict[int, int] = {}  # tid -> di
        matched_dets: Dict[int, int] = {}  # di -> tid
        for iou, tid, di in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks[tid] = di
            matched_dets[di] = tid
            det_used[di] = True
            if self.config.log_assignments:
                logger.info("match track_id=%s det=%s iou=%.2f", tid, di, iou)

        # Update matched tracks.
        for tid, di in matched_tracks.items():
            tr = self._tracks[tid]
            tr.bbox = det_bboxes[di]
            tr.last_seen_ts = now
            tr.hits += 1
            tr.misses = 0

        # Age unmatched tracks and prune old ones.
        to_delete: List[int] = []
        for tid, tr in self._tracks.items():
            if tid in matched_tracks:
                continue
            tr.misses += 1
            if (now - tr.last_seen_ts) > self.config.max_age_seconds:
                to_delete.append(tid)
        for tid in to_delete:
            if self.config.log_assignments:
                logger.info("drop track_id=%s age=%.2fs", tid, now - self._tracks[tid].last_seen_ts)
            self._tracks.pop(tid, None)

        # Create new tracks for unmatched detections.
        for di, used in enumerate(det_used):
            if used:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(track_id=tid, bbox=det_bboxes[di], last_seen_ts=now)
            if self.config.log_assignments:
                logger.info("new track_id=%s det=%s", tid, di)

        # Emit confirmed tracks as tracked objects.
        out: List[TrackedObject] = []
        for tid, tr in self._tracks.items():
            if tr.hits < self.config.min_hits_to_confirm:
                continue
            x1, y1, x2, y2 = tr.bbox
            out.append((x1, y1, x2, y2, tid))
        return out

