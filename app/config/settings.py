from __future__ import annotations

from typing import Literal, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Centralized configuration loaded from environment variables.

    This keeps runtime configuration deployment-friendly (Docker/K8s/etc).
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Runtime
    stub_mode: bool = Field(default=False, alias="STUB_MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Paths
    video_path: str = Field(default="data/parking_video.mp4", alias="VIDEO_PATH")
    slots_path: str = Field(default="app/slots/slots.json", alias="SLOTS_PATH")
    model_path: str = Field(default="models/yolov8n.pt", alias="MODEL_PATH")

    # Pipeline performance
    frame_stride: int = Field(default=3, alias="FRAME_STRIDE")
    metrics_period_s: float = Field(default=2.0, alias="METRICS_PERIOD_S")
    analytics_period_s: float = Field(default=2.0, alias="ANALYTICS_PERIOD_S")

    # Detection
    detector_backend: Literal["pytorch", "onnx"] = Field(default="pytorch", alias="DETECTOR_BACKEND")
    detection_conf: float = Field(default=0.3, alias="DETECTION_CONF")
    vehicle_class_ids: Tuple[int, ...] = (2, 3, 5, 7)

    # Tracking (IoU tracker)
    tracker_iou_threshold: float = Field(default=0.3, alias="TRACKER_IOU_THRESHOLD")
    tracker_max_age_s: float = Field(default=1.5, alias="TRACKER_MAX_AGE_S")
    tracker_min_hits: int = Field(default=1, alias="TRACKER_MIN_HITS")
    tracker_log_assignments: bool = Field(default=False, alias="TRACKER_LOG_ASSIGNMENTS")

    # Occupancy smoothing
    debounce_frames: int = Field(default=4, alias="DEBOUNCE_FRAMES")
    clear_timeout_s: float = Field(default=1.5, alias="CLEAR_TIMEOUT_S")

    # Analytics / Alerts
    busiest_slots_k: int = Field(default=3, alias="BUSIEST_SLOTS_K")
    overstay_threshold_s: float = Field(default=90.0, alias="OVERSTAY_THRESHOLD_S")
    almost_full_threshold: float = Field(default=0.10, alias="ALMOST_FULL_THRESHOLD")


def get_settings() -> AppSettings:
    return AppSettings()

