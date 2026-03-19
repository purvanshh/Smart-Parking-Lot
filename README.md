# Smart Parking Occupancy & Availability API

Production-style computer vision backend that converts parking-lot video into stable occupancy state, durations, and operational analytics through FastAPI.

## Problem this solves

Parking operations need reliable slot-level visibility, not just raw detections. Real deployments fail when outputs flicker, IDs switch, or occlusions break logic.

This system focuses on **robust reasoning over time**:
- detects and tracks vehicles,
- maps them to slot polygons,
- stabilizes occupancy with temporal rules,
- exposes actionable API outputs for operations.

## System architecture

```text
Video feed
  -> YOLOv8 detection
  -> IoU multi-object tracker
  -> slot mapping (polygon containment)
  -> occupancy engine (debounce + occlusion handling + durations)
  -> analytics + alerts
  -> FastAPI endpoints
```

Core principle: this is not only detection, it is a **stateful real-time system**.

## Engineering decisions

- **Temporal smoothing**: slot state changes are debounced to prevent flicker.
- **Occlusion handling**: slots are not cleared immediately when detections drop.
- **IoU tracker tradeoff**: lightweight and fast for demo/internship scope; heavier trackers (e.g. ByteTrack) can improve difficult scenes.
- **Operational outputs**: API serves occupancy, durations, sessions, and alerts.

## Features

- Real-time pipeline: video -> detection -> tracking -> occupancy
- Stable slot status with debounce + timeout logic
- Entry/exit timestamps and parking duration per track
- Analytics: utilization, busiest/least-used slots
- Alerts: overstay and lot almost full
- Graceful idle mode if model/video is unavailable (API stays up)

## API reference

Base URL: `http://127.0.0.1:8000`

### `GET /health`

Returns runtime status and pipeline health.

```json
{
  "status": "ok",
  "pipeline_status": "running",
  "video_loaded": true,
  "model_loaded": true,
  "last_frame_ts": 1710000000.12,
  "fps_estimate": 19.84,
  "active_tracks": 2
}
```

### `GET /slots`

Stable slot state:

```json
{
  "A1": "occupied",
  "A2": "empty"
}
```

### `GET /summary`

```json
{
  "total_slots": 2,
  "occupied": 1,
  "available": 1
}
```

### `GET /analytics`

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

### `GET /sessions?limit=50`

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

```json
[
  {
    "type": "overstay",
    "timestamp": 1710000212.1,
    "metadata": { "track_id": 31, "slot_id": "A2", "duration_s": 121.7 }
  }
]
```

## Setup

### 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Data and model

- Place video in `data/` (default: `data/parking_video.MOV`)
- Slot polygons in `app/slots/slots.json`
- Model defaults to `yolov8n.pt` (auto-download on first run)
- Data link (not committed): [OneDrive dataset](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBanVMOExkRmtDaFppT1ZpcjB2Uk1QUGpMdUZUV2c_ZT1oaTJoZTM&id=59289045B7F08B3B!144101&cid=59289045B7F08B3B&sb=name&sd=1)

### 3) One-command run

```bash
uvicorn main:app --reload
```

### 4) Quick verification

```bash
python3 -c "import cv2; cap=cv2.VideoCapture('data/parking_video.MOV'); print('opened=', cap.isOpened())"
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/slots
```

## Configuration

All runtime knobs are centralized in `app/config/settings.py` and exposed via env vars:

- `MODEL_PATH`
- `VIDEO_PATH`
- `DETECTION_CONF`
- `FRAME_STRIDE`
- `TRACKER_MAX_AGE_S`
- `DEBOUNCE_FRAMES`
- `DEMO_MODE`
- `ANALYTICS_PERIOD_S`
- `METRICS_PERIOD_S`

Example:

```bash
DEMO_MODE=1 FRAME_STRIDE=1 DETECTION_CONF=0.05 TRACKER_MAX_AGE_S=4.0 DEBOUNCE_FRAMES=8 uvicorn main:app --reload
```

## Demo walkthrough (60-90 seconds)

### Commands

```bash
source .venv/bin/activate
uvicorn main:app --reload
```

### What to show

1. Terminal logs: `event=video_opened`, `event=model_loaded`, periodic `event=pipeline_summary`
2. `GET /health` to show FPS, active tracks, loaded status
3. `GET /slots` and `GET /summary` while video is running
4. `GET /analytics`, `GET /sessions`, `GET /alerts`

### What to say (short script)

- “This is a real-time backend system, not just a detector.”
- “We detect + track vehicles, then apply temporal logic to produce stable occupancy.”
- “Debounce and occlusion tolerance prevent flicker and false transitions.”
- “The API serves both live status and operational insights like utilization and overstay alerts.”

## Limitations

- Fixed-camera and manual slot calibration.
- IoU tracker is lightweight; crowded scenes may benefit from stronger MOT.
- In-memory persistence only (single-process scope).

## Repo structure

```text
app/
  api/
  analytics/
  config/
  detection/
  profiling/
  tracking/
  occupancy/
  slots/
data/
models/
main.py
requirements.txt
```

`data/` and large model binaries are gitignored to keep the repository clean.

