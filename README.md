<p align="center">
  <h1 align="center">Smart Parking Lot</h1>
  <p align="center">
    <strong>Real-time parking occupancy detection & analytics powered by computer vision</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#getting-started">Getting Started</a> ·
    <a href="#api-reference">API Reference</a> ·
    <a href="#configuration">Configuration</a>
  </p>
</p>

---

## Overview

Smart Parking Lot is a **production-grade computer vision backend** that transforms raw parking lot video into stable, slot-level occupancy state, parking durations, and operational analytics — all served through a FastAPI REST API.

Unlike simple detection demos, this system is built as a **stateful real-time pipeline** that addresses the core challenges of real-world deployment: detection flicker, identity switching, and occlusion handling. Every slot status change is temporally smoothed, every vehicle is tracked with a persistent ID, and every transition is debounced before it reaches the API consumer.

---

## Problem Statement

Parking management systems require **reliable slot-level visibility**, not just raw object detections. Real deployments fail when:

- 🔴 Detections flicker frame-to-frame, producing false vacancy/occupancy transitions
- 🔴 Vehicle IDs switch across frames, corrupting session and duration tracking  
- 🔴 Brief occlusions (shadows, pedestrians) incorrectly clear occupied slots  
- 🔴 There is no distinction between transient sensor noise and genuine state changes

This system solves all of the above through a multi-stage pipeline with **temporal reasoning, debounce logic, and occlusion-tolerant occupancy management**.

---

## Features

| Category | Capability |
|---|---|
| **Detection** | YOLOv8-based vehicle detection with configurable confidence thresholds and COCO class filtering (car, motorcycle, bus, truck) |
| **Tracking** | Greedy IoU multi-object tracker with stable persistent IDs, occlusion tolerance, and configurable max-age timeout |
| **Occupancy** | Polygon-based slot assignment with debounced state transitions and clear-timeout for occlusion resilience |
| **Sessions** | Per-vehicle entry/exit timestamps, parking duration tracking, and completed session history |
| **Analytics** | Real-time slot utilization rates, busiest/least-used slot rankings, and average parking durations |
| **Alerts** | Overstay detection and lot-almost-full warnings with configurable thresholds |
| **Resilience** | Graceful idle mode — API remains available even when video/model is unavailable |
| **Profiling** | Built-in pipeline profiler with per-stage latency (EMA-smoothed) and real-time FPS estimation |
| **Configuration** | 25+ runtime parameters, all configurable via environment variables (Docker/K8s-ready) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Smart Parking Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Video Feed ──► YOLOv8 Detector ──► IoU Tracker ──► Slot Mapper   │
│                   (detector.py)      (tracker.py)   (slot_manager)  │
│                                                         │           │
│                                                         ▼           │
│                                              Occupancy Engine       │
│                                              (debounce + timeout    │
│                                               + session tracking)   │
│                                                         │           │
│                                          ┌──────────────┼────────┐  │
│                                          ▼              ▼        │  │
│                                   Analytics Engine    FastAPI     │  │
│                                   (utilization,       REST API   │  │
│                                    alerts, stats)     (routes)   │  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| **1. Detection** | `app/detection/detector.py` | Wraps Ultralytics YOLOv8 with lazy loading. Filters output to vehicle classes only (COCO IDs 2, 3, 5, 7) and applies confidence thresholding. |
| **2. Tracking** | `app/tracking/tracker.py` | Greedy IoU-based multi-object tracker. Produces stable `track_id`s across frames, tolerates brief occlusion via `max_age_seconds`, and supports a `min_hits` confirmation gate. |
| **3. Slot Mapping** | `app/slots/slot_manager.py` | Maps vehicle center-points to parking slot polygons using a ray-casting point-in-polygon algorithm (`app/utils/geometry.py`). Slot definitions are loaded from a JSON configuration file. |
| **4. Occupancy** | `app/occupancy/engine.py` | Core state machine. Debounces transitions (N consecutive frames required), applies a clear-timeout for occlusion tolerance, tracks entry/exit timestamps, and finalizes parking sessions on vehicle departure. |
| **5. Analytics** | `app/analytics/engine.py` | Incrementally ingests completed sessions (no reprocessing). Computes utilization rates, average durations, busiest-slot rankings and live alerts (overstay, lot-almost-full). |
| **6. API** | `app/api/routes.py` | Exposes 6 read-only FastAPI endpoints (`/health`, `/slots`, `/summary`, `/analytics`, `/sessions`, `/alerts`). |

### Key Engineering Decisions

- **Temporal Smoothing** — Slot state changes are debounced across multiple consecutive frames to eliminate flicker from noisy detections.
- **Occlusion Tolerance** — Occupied slots are not cleared immediately when detections drop. A configurable `clear_timeout_s` window prevents brief occlusions from triggering false vacancy.
- **Lightweight Tracker** — The IoU tracker is intentionally simple and fast, prioritizing low-latency operation. The interface is ByteTrack-compatible for drop-in upgrades.
- **Graceful Degradation** — If the model or video fails to load, the system automatically enters idle mode where the API remains responsive with stale/empty data.
- **Demo Mode** — A dedicated mode with higher debounce, detection reuse, and short-lived track filtering for stable live demonstrations.

---

## Getting Started

### Prerequisites

- **Python** 3.10+
- **pip** (or any Python package manager)
- A parking lot video file (`.MOV`, `.mp4`, etc.)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/purvanshh/Smart-Parking-Lot.git
cd Smart-Parking-Lot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data & Model

| Asset | Location | Notes |
|---|---|---|
| Parking video | `data/parking_video.MOV` | Place your video here. A sample dataset is available via [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBanVMOExkRmtDaFppT1ZpcjB2Uk1QUGpMdUZUV2c_ZT1oaTJoZTM&id=59289045B7F08B3B!144101&cid=59289045B7F08B3B&sb=name&sd=1). |
| YOLOv8 model | `models/yolov8n.pt` | Auto-downloaded on first run via Ultralytics. No manual setup needed. |
| Slot polygons | `app/slots/slots.json` | Define your parking slot boundaries as polygon coordinates. |

### 3. Run the Server

```bash
uvicorn main:app --reload
```

The API will be available at **`http://127.0.0.1:8000`**.

### 4. Verify

```bash
# Check pipeline health
curl http://127.0.0.1:8000/health

# Check slot statuses
curl http://127.0.0.1:8000/slots

# Check occupancy summary
curl http://127.0.0.1:8000/summary
```

---

## API Reference

**Base URL:** `http://127.0.0.1:8000`

### `GET /health`

Returns pipeline runtime status, model/video load state, FPS estimate, and active track count.

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

Returns stable occupancy state for each parking slot (debounced, not raw detections).

```json
{
  "A1": "occupied",
  "A2": "empty"
}
```

### `GET /summary`

Returns aggregate slot counts.

```json
{
  "total_slots": 2,
  "occupied": 1,
  "available": 1
}
```

### `GET /analytics`

Returns computed insights: total vehicles served, average durations, per-slot utilization rates, and busiest/least-used rankings.

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

Returns recent completed parking sessions with track ID, slot ID, timestamps, and duration.

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

Returns active operational alerts (overstay detections, lot-almost-full warnings).

```json
[
  {
    "type": "overstay",
    "timestamp": 1710000212.1,
    "metadata": {
      "track_id": 31,
      "slot_id": "A2",
      "duration_s": 121.7
    }
  }
]
```

---

## Configuration

All parameters are managed via **environment variables** through `pydantic-settings`, making the system deployment-friendly for Docker, Kubernetes, or any CI/CD environment.

### Runtime

| Variable | Default | Description |
|---|---|---|
| `STUB_MODE` | `false` | Run without model/video (API-only mode) |
| `DEMO_MODE` | `false` | Enable demo stabilization (higher debounce, detection reuse) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Paths

| Variable | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `data/parking_video.MOV` | Input video file |
| `SLOTS_PATH` | `app/slots/slots.json` | Slot polygon definitions |
| `MODEL_PATH` | `yolov8n.pt` | YOLOv8 model weights |

### Detection & Tracking

| Variable | Default | Description |
|---|---|---|
| `DETECTION_CONF` | `0.3` | Minimum confidence threshold |
| `FRAME_STRIDE` | `3` | Process every Nth frame |
| `TRACKER_IOU_THRESHOLD` | `0.3` | Minimum IoU for track association |
| `TRACKER_MAX_AGE_S` | `1.5` | Seconds before unmatched track is dropped |
| `TRACKER_MIN_HITS` | `1` | Minimum hits to confirm a track |

### Occupancy & Analytics

| Variable | Default | Description |
|---|---|---|
| `DEBOUNCE_FRAMES` | `4` | Consecutive frames required to commit a state change |
| `CLEAR_TIMEOUT_S` | `1.5` | Grace period before clearing an undetected vehicle |
| `OVERSTAY_THRESHOLD_S` | `90.0` | Duration (seconds) after which an overstay alert fires |
| `ALMOST_FULL_THRESHOLD` | `0.10` | Available/total ratio below which a lot-almost-full alert fires |
| `ANALYTICS_PERIOD_S` | `2.0` | Analytics recomputation interval |

### Example: Custom Launch

```bash
DEMO_MODE=1 \
FRAME_STRIDE=1 \
DETECTION_CONF=0.05 \
TRACKER_MAX_AGE_S=4.0 \
DEBOUNCE_FRAMES=8 \
uvicorn main:app --reload
```

---

## Project Structure

```
Smart-Parking-Lot/
├── main.py                          # Application entrypoint & pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── app/
│   ├── api/
│   │   └── routes.py                # FastAPI endpoint definitions
│   ├── analytics/
│   │   └── engine.py                # Incremental analytics & alert computation
│   ├── config/
│   │   └── settings.py              # Centralized env-var configuration (pydantic-settings)
│   ├── detection/
│   │   └── detector.py              # YOLOv8 detector wrapper
│   ├── occupancy/
│   │   └── engine.py                # Stateful occupancy engine (debounce, sessions)
│   ├── profiling/
│   │   └── metrics.py               # Per-stage latency & FPS profiler
│   ├── slots/
│   │   ├── slot_manager.py          # Polygon-based slot assignment
│   │   └── slots.json               # Parking slot polygon definitions
│   ├── tracking/
│   │   └── tracker.py               # IoU-based multi-object tracker
│   └── utils/
│       └── geometry.py              # Ray-casting point-in-polygon algorithm
├── data/                            # Video files (gitignored)
└── models/                          # Model weights (gitignored)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **Object Detection** | [Ultralytics YOLOv8](https://docs.ultralytics.com/) |
| **Computer Vision** | [OpenCV](https://opencv.org/) |
| **Configuration** | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| **Inference Runtime** | [ONNX Runtime](https://onnxruntime.ai/) (optional) |
| **Data Processing** | [NumPy](https://numpy.org/) |
| **Annotation** | [Supervision](https://supervision.roboflow.com/) |

---

## Limitations & Future Work

| Current Limitation | Potential Improvement |
|---|---|
| Fixed-camera, manual slot polygon calibration | Auto-calibration via perspective transforms or segmentation |
| Greedy IoU tracker — may struggle in dense scenes | Upgrade to ByteTrack or Deep SORT for stronger association |
| In-memory persistence (single-process scope) | Add Redis/PostgreSQL for persistence and multi-worker support |
| No authentication on API endpoints | Add API key or OAuth2 middleware |
| Single video input | Support RTSP streams and multi-camera setups |

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/purvanshh">purvanshh</a>
</p>
