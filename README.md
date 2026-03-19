# Smart Parking Occupancy & Availability API

Real-time computer vision pipeline that turns a parking-lot video feed into **stable slot occupancy**, **parking durations**, and **operational analytics** via a clean FastAPI interface.

## Problem statement (real-world)

Parking facilities often lack real-time visibility into which slots are occupied. This increases driver search time, reduces space utilization, and makes operational decisions reactive instead of data-driven.

This project detects and tracks vehicles from a fixed-camera video feed, maps them to predefined slot polygons, and serves **live occupancy + analytics** through an API.

## Key features (what makes this interview-ready)

- **Real-time pipeline**: Video → YOLOv8 detection → tracking → slot mapping → occupancy → API
- **Robust occupancy**: debouncing + occlusion tolerance to prevent flicker
- **Time-based reasoning**: entry/exit events and per-vehicle **parking duration**
- **Decision support**: utilization analytics + rule-based alerts (overstay, lot almost full)
- **Modular design**: detection / tracking / occupancy / analytics are decoupled

## System architecture (high level)

```text
Video (OpenCV) → Detection (YOLOv8) → Tracking (MOT IDs) → Slot mapping (polygons)
→ Occupancy engine (debounce + duration) → Analytics/alerts → FastAPI
```

### Core modules

- **Detector**: vehicle-only detection (car/truck/bus/motorcycle), configurable confidence threshold
- **Tracker**: stable `track_id` across frames (IoU-based; ByteTrack-ready interface)
- **SlotManager**: assigns box center points to slot polygons (`app/slots/slots.json`)
- **OccupancyEngine**: stable slot state, entry/exit/duration, completed sessions
- **AnalyticsEngine**: utilization, busiest/least-used slots, overstay + almost-full alerts

## Tech stack

- **Python**, **FastAPI**
- **YOLOv8 (Ultralytics)**, **OpenCV**
- **Tracking**: IoU-based MOT (ByteTrack-ready interface)
- **Storage**: in-memory state (sessions + analytics snapshots)
- **Optional**: ONNX Runtime path, Docker container

## API

Base URL: `http://127.0.0.1:8000`

### `GET /health`

Returns liveness + pipeline timing signal.

Example response:

```json
{
  "status": "ok",
  "last_update_ts": 1710000000.12
}
```

### `GET /slots`

Slot-level stable occupancy state (debounced, occlusion-tolerant).

Example response:

```json
{
  "A1": "occupied",
  "A2": "empty"
}
```

### `GET /summary`

Aggregate counts.

Example response:

```json
{
  "total_slots": 2,
  "occupied": 1,
  "available": 1
}
```

### `GET /analytics`

Operational insights computed periodically (not every frame).

Example response:

```json
{
  "total_vehicles": 27,
  "avg_duration_s": 84.6,
  "utilization": { "A1": 0.62, "A2": 0.31 },
  "busiest_slots": ["A1", "A2"],
  "least_used_slots": ["A2", "A1"],
  "avg_duration_per_slot_s": { "A1": 92.3, "A2": 73.1 }
}
```

Field notes:
- `utilization[slot]`: \(occupied\_time / total\_time\) over the analytics window since the engine started (includes active vehicles best-effort).

### `GET /sessions?limit=50`

Recent completed parking sessions (most recent last).

Example response:

```json
[
  {
    "track_id": 12,
    "slot_id": "A1",
    "entry_ts": 1710000100.0,
    "exit_ts": 1710000188.4,
    "duration_s": 88.4
  }
]
```

### `GET /alerts`

Active alerts derived from current occupancy + durations.

Example response:

```json
[
  {
    "type": "lot_almost_full",
    "timestamp": 1710000200.5,
    "metadata": { "available": 1, "total": 10, "ratio": 0.1 }
  },
  {
    "type": "overstay",
    "timestamp": 1710000212.1,
    "metadata": { "track_id": 31, "slot_id": "A2", "duration_s": 121.7 }
  }
]
```

## Setup

### 1) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Provide inputs

- **Video**: `data/parking_video.mp4`
- **Slots**: `app/slots/slots.json` (slot_id → polygon vertices)
- **Model**: `models/yolov8n.pt` (or set `MODEL_PATH` to another YOLOv8 weight file)

### 4) Run the API

```bash
uvicorn main:app --reload
```

## Configuration (deployment-friendly)

All knobs are environment-driven via `app/config/settings.py`. Common ones:

- `FRAME_STRIDE=3` (process every 3rd frame)
- `DETECTION_CONF=0.3`
- `DEBOUNCE_FRAMES=4`
- `CLEAR_TIMEOUT_S=1.5`
- `OVERSTAY_THRESHOLD_S=90`
- `ANALYTICS_PERIOD_S=2.0`

Example:

```bash
FRAME_STRIDE=3 DETECTION_CONF=0.35 OVERSTAY_THRESHOLD_S=120 uvicorn main:app --reload
```

## Demo expectations (what to look for)

- `/slots` is **stable** (no flicker) even with brief occlusions
- `/sessions` accumulates completed sessions with durations
- `/analytics` evolves over time (utilization + busiest slots)
- `/alerts` triggers when rules match (overstay / almost-full)

## Repo structure

```text
app/
  api/routes.py
  analytics/engine.py
  config/settings.py
  detection/detector.py
  tracking/tracker.py
  occupancy/engine.py
  slots/slot_manager.py
  slots/slots.json
  utils/geometry.py
data/
models/
main.py
requirements.txt
README.md
```

## Limitations (intentional, honest)

- Fixed-camera assumption; slot polygons are manually configured.
- Tracker is IoU-based (fast + simple); ByteTrack can improve re-identification under heavy occlusion.
- In-memory state only (no DB). Suitable for a single-process demo; production would persist sessions/metrics.

## Future improvements (credible next steps)

- Replace IoU tracker with ByteTrack + tuned parameters for dense scenes.
- Add persistent storage for sessions + analytics (SQLite/Postgres).
- Add a lightweight dashboard (optional) or Prometheus metrics export.

