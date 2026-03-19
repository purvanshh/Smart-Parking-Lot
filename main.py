from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI

from app.api.routes import build_router
from app.analytics.engine import AnalyticsEngine
from app.config.settings import AppSettings, get_settings
from app.detection.detector import Detector, DetectorConfig
from app.occupancy.engine import OccupancyEngine, TrackedObject
from app.profiling.metrics import PipelineProfiler
from app.slots.slot_manager import SlotManager
from app.tracking.tracker import Tracker, TrackerConfig


settings: AppSettings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("smart_parking")

occupancy_engine = OccupancyEngine()
slot_manager = SlotManager.from_json(Path(settings.slots_path))
occupancy_engine.bootstrap_slots(slot_manager)
occupancy_engine.clear_timeout_s = settings.clear_timeout_s
occupancy_engine.debounce_frames = max(settings.debounce_frames, 8) if settings.demo_mode else settings.debounce_frames

tracker = Tracker(
    TrackerConfig(
        iou_threshold=settings.tracker_iou_threshold,
        max_age_seconds=settings.tracker_max_age_s,
        min_hits_to_confirm=settings.tracker_min_hits,
        log_assignments=settings.tracker_log_assignments and (not settings.demo_mode),
    )
)
detector = None
analytics_engine = AnalyticsEngine(
    occupancy_engine=occupancy_engine,
    busiest_k=settings.busiest_slots_k,
    overstay_threshold_s=settings.overstay_threshold_s,
    almost_full_threshold=settings.almost_full_threshold,
)


@dataclass
class RuntimeState:
    pipeline_status: str = "starting"
    video_loaded: bool = False
    model_loaded: bool = False
    last_frame_ts: float | None = None
    fps_estimate: float = 0.0
    active_tracks: int = 0


runtime_state = RuntimeState()
runtime_lock = threading.Lock()
profiler = PipelineProfiler(fps_window_size=30)


def _health_payload() -> Dict:
    with runtime_lock:
        return {
            "status": "ok",
            "pipeline_status": runtime_state.pipeline_status,
            "video_loaded": runtime_state.video_loaded,
            "model_loaded": runtime_state.model_loaded,
            "last_frame_ts": runtime_state.last_frame_ts,
            "fps_estimate": round(runtime_state.fps_estimate, 3),
            "active_tracks": runtime_state.active_tracks,
        }


app = FastAPI(title="Smart Parking Occupancy & Availability API")
app.include_router(build_router(occupancy_engine, analytics_engine, _health_payload))


def pipeline_loop() -> None:
    """
    Background processing loop with graceful idle fallback.
    """
    cap = None
    idle_mode = settings.stub_mode
    if not settings.stub_mode:
        import cv2

        global detector
        try:
            detector = Detector(
                DetectorConfig(
                    model_path=settings.model_path,
                    conf=settings.detection_conf,
                    vehicle_class_ids=settings.vehicle_class_ids,
                )
            )
            with runtime_lock:
                runtime_state.model_loaded = True
            logger.info("event=model_loaded path=%s conf=%.3f", settings.model_path, settings.detection_conf)
        except Exception:
            idle_mode = True
            logger.exception("event=model_load_failed path=%s mode=idle", settings.model_path)

        cap = cv2.VideoCapture(str(Path(settings.video_path)))
        if cap.isOpened():
            with runtime_lock:
                runtime_state.video_loaded = True
            logger.info("event=video_opened path=%s", settings.video_path)
        else:
            idle_mode = True
            logger.warning("event=video_open_failed path=%s mode=idle", settings.video_path)
    else:
        logger.info("event=stub_mode_enabled mode=idle")

    frame_idx = 0
    last_analytics_ts = 0.0
    last_metrics_ts = 0.0
    last_detections: List = []
    last_detection_frame = -10_000
    track_first_seen_ts: Dict[int, float] = {}

    with runtime_lock:
        runtime_state.pipeline_status = "idle" if idle_mode else "running"

    while True:
        try:
            if idle_mode:
                now_ts = time.time()
                occupancy_engine.update([], slot_manager, now_ts=now_ts)
                if (now_ts - last_analytics_ts) >= settings.analytics_period_s:
                    analytics_engine.update(now_ts=now_ts)
                    last_analytics_ts = now_ts
                with runtime_lock:
                    runtime_state.last_frame_ts = now_ts
                    runtime_state.fps_estimate = profiler.fps()
                    runtime_state.active_tracks = 0
                time.sleep(0.5)
                continue

            import cv2

            assert cap is not None
            ok, frame = cap.read()
            if not ok:
                # Loop video stream gracefully.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1
            if settings.frame_stride > 1 and (frame_idx % settings.frame_stride) != 0:
                continue

            assert detector is not None
            now_ts = time.time()
            det_t0 = time.perf_counter()
            detections = detector.detect(frame)
            profiler.update_stage_ms("detect", (time.perf_counter() - det_t0) * 1000.0)

            if settings.demo_mode:
                # Demo stability: reuse last detections for up to 2 processed frames.
                if detections:
                    last_detections = detections
                    last_detection_frame = frame_idx
                elif last_detections and (frame_idx - last_detection_frame) <= 2:
                    detections = last_detections

            trk_t0 = time.perf_counter()
            tracked_objects: List[TrackedObject] = tracker.update(detections, now_ts=now_ts)
            profiler.update_stage_ms("track", (time.perf_counter() - trk_t0) * 1000.0)

            if settings.demo_mode:
                # Ignore very short-lived tracks to reduce flicker in demos.
                filtered: List[TrackedObject] = []
                for obj in tracked_objects:
                    tid = obj[4]
                    first_seen = track_first_seen_ts.setdefault(tid, now_ts)
                    if (now_ts - first_seen) >= max(2.0, settings.min_track_age_s):
                        filtered.append(obj)
                tracked_objects = filtered

            occ_t0 = time.perf_counter()
            occupancy_engine.update(tracked_objects, slot_manager, now_ts=now_ts)
            profiler.update_stage_ms("occupancy", (time.perf_counter() - occ_t0) * 1000.0)
            profiler.mark_frame(now_ts)

            if (now_ts - last_analytics_ts) >= settings.analytics_period_s:
                analytics_engine.update(now_ts=now_ts)
                last_analytics_ts = now_ts

            fps = profiler.fps()
            summary = occupancy_engine.summary()
            with runtime_lock:
                runtime_state.last_frame_ts = now_ts
                runtime_state.fps_estimate = fps
                runtime_state.active_tracks = len(tracked_objects)

            if (now_ts - last_metrics_ts) >= settings.metrics_period_s:
                logger.info(
                    "event=pipeline_summary frame=%s fps=%.2f detections=%s tracks=%s occupied=%s total_slots=%s detect_ms=%.3f track_ms=%.3f occupancy_ms=%.3f demo_mode=%s",
                    frame_idx,
                    fps,
                    len(detections),
                    len(tracked_objects),
                    summary["occupied"],
                    summary["total_slots"],
                    profiler.stage_summary_ms().get("detect", 0.0),
                    profiler.stage_summary_ms().get("track", 0.0),
                    profiler.stage_summary_ms().get("occupancy", 0.0),
                    settings.demo_mode,
                )
                last_metrics_ts = now_ts

            time.sleep(0.001)
        except Exception:
            logger.exception("Pipeline loop error; continuing after brief backoff")
            time.sleep(1.0)


@app.on_event("startup")
def _startup() -> None:
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()

