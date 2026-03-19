# Smart Parking Occupancy & Availability API (Starter)

Backend-only starter scaffold for a real-time parking occupancy system:

Frame → Detection → Tracking → Center Point → Slot Mapping → Occupancy Update → API Response

## What’s included (starter boilerplate)

- FastAPI server with:
  - `GET /health`
  - `GET /slots`
  - `GET /summary`
- Background “pipeline loop” thread (stubbed by default so it runs immediately)
- Slot polygon config via `app/slots/slots.json`
- Core modules wired up: detector, tracker, geometry, slot manager, occupancy engine

## Quickstart

### 1) Create venv + install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the API

```bash
uvicorn main:app --reload
```

Open:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/slots`
- `http://127.0.0.1:8000/summary`

## How the “pipeline” works in this starter

By default, the pipeline runs in **stub mode** (no video/model required) and just exercises the occupancy engine with empty detections so the API boots cleanly.

When you’re ready:
- Put your video at `data/parking_video.mp4`
- Put your YOLO weights at `models/yolov8n.pt` (or use the default Ultralytics model name)
- Update `main.py` config to `STUB_MODE = False`

## Repo structure

```text
app/
  api/routes.py
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

