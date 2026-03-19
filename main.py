from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI

from app.api.routes import build_router
from app.detection.detector import Detector, DetectorConfig
from app.occupancy.engine import OccupancyEngine, TrackedObject
from app.slots.slot_manager import SlotManager
from app.tracking.tracker import Tracker

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("smart_parking")

# If you need to boot the API without model/video, set this to True.
STUB_MODE = os.getenv("STUB_MODE", "0") == "1"

# Performance knob: process every Nth frame (e.g. 3 = every 3rd frame).
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "3"))

SLOTS_PATH = Path("app/slots/slots.json")
VIDEO_PATH = Path("data/parking_video.mp4")
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")


occupancy_engine = OccupancyEngine()
slot_manager = SlotManager.from_json(SLOTS_PATH)
occupancy_engine.bootstrap_slots(slot_manager)

tracker = Tracker()
detector = None

app = FastAPI(title="Smart Parking Occupancy & Availability API")
app.include_router(build_router(occupancy_engine))


def pipeline_loop() -> None:
    """
    Background pipeline loop (runs off the FastAPI main thread).

    Steps:
    - Read frames from a local video file using OpenCV `VideoCapture`
    - Loop the video when it ends
    - Run YOLOv8 detection (vehicle classes only, conf >= 0.3)
    - Apply a temporary tracker (IDs per detection per frame)
    - Map each tracked box center to a parking slot polygon
    - Update `OccupancyEngine.slot_status`, which drives `/slots` and `/summary`
    """
    cap = None
    if not STUB_MODE:
        import cv2

        # Load model inside the background thread to avoid slowing FastAPI startup.
        global detector
        detector = Detector(DetectorConfig(model_path=MODEL_PATH, conf=0.3))

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video at {VIDEO_PATH}")
        logger.info("Video opened: %s", VIDEO_PATH)
        logger.info("Detector loaded: %s (conf>=0.3, vehicle classes only)", MODEL_PATH)
        logger.info("Frame stride: %s", FRAME_STRIDE)
    else:
        logger.warning("STUB_MODE enabled: pipeline will not run detection/video.")

    frame_idx = 0
    while True:
        try:
            if STUB_MODE:
                # Keeps API stable and predictable when booting without runtime deps.
                occupancy_engine.update([], slot_manager)
                time.sleep(0.5)
                continue

            import cv2

            assert cap is not None
            ok, frame = cap.read()
            if not ok:
                # Loop the video when it ends or on decode hiccups.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1
            if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE) != 0:
                continue

            assert detector is not None
            now_ts = time.time()
            detections = detector.detect(frame)
            tracked_objects: List[TrackedObject] = tracker.update(detections, now_ts=now_ts)
            occupancy_engine.update(tracked_objects, slot_manager, now_ts=now_ts)

            logger.info(
                "Processed frame=%s detections=%s tracked=%s occupied=%s/%s",
                frame_idx,
                len(detections),
                len(tracked_objects),
                occupancy_engine.summary()["occupied"],
                occupancy_engine.summary()["total_slots"],
            )

            # Small sleep to avoid pegging CPU; video FPS will still dominate.
            time.sleep(0.001)
        except Exception:
            logger.exception("Pipeline loop error; continuing after brief backoff")
            time.sleep(1.0)


@app.on_event("startup")
def _startup() -> None:
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()

