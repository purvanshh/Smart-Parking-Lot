from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class PipelineProfiler:
    """
    Lightweight runtime profiler for per-stage latency and FPS.
    """

    fps_window_size: int = 30
    _frame_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _stage_ms_ema: Dict[str, float] = field(default_factory=dict)  # stage -> ms
    _ema_alpha: float = 0.2

    def mark_frame(self, ts: float | None = None) -> None:
        self._frame_timestamps.append(time.time() if ts is None else ts)

    def update_stage_ms(self, stage: str, value_ms: float) -> None:
        prev = self._stage_ms_ema.get(stage)
        if prev is None:
            self._stage_ms_ema[stage] = value_ms
            return
        self._stage_ms_ema[stage] = (self._ema_alpha * value_ms) + ((1.0 - self._ema_alpha) * prev)

    def fps(self) -> float:
        if len(self._frame_timestamps) < 2:
            return 0.0
        dt = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if dt <= 1e-9:
            return 0.0
        return (len(self._frame_timestamps) - 1) / dt

    def stage_summary_ms(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self._stage_ms_ema.items()}

