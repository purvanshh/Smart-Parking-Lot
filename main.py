from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI

from app.api.routes import build_router
from app.occupancy.engine import OccupancyEngine, TrackedObject
from app.slots.slot_manager import SlotManager
from app.tracking.tracker import Tracker

# Toggle this when you’re ready to connect real detection/video.
STUB_MODE = True

SLOTS_PATH = Path("app/slots/slots.json")
VIDEO_PATH = Path("data/parking_video.mp4")


occupancy_engine = OccupancyEngine()
slot_manager = SlotManager.from_json(SLOTS_PATH)
occupancy_engine.bootstrap_slots(slot_manager)

tracker = Tracker()

app = FastAPI(title="Smart Parking Occupancy & Availability API")
app.include_router(build_router(occupancy_engine))


def pipeline_loop() -> None:
    """
    Starter pipeline loop.
    - In stub mode: updates occupancy with empty tracks.
    - In non-stub mode: reads frames from VIDEO_PATH and (for now) still uses empty detections
      until you wire the detector into the loop.
    """
    cap = None
    if not STUB_MODE:
        import cv2

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video at {VIDEO_PATH}")

    while True:
        if cap is not None:
            import cv2

            ok, _frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # Starter: no detections yet (keeps API stable & testable)
        tracked_objects: List[TrackedObject] = tracker.update([])
        occupancy_engine.update(tracked_objects, slot_manager)

        time.sleep(0.2)  # 5 Hz


@app.on_event("startup")
def _startup() -> None:
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()

