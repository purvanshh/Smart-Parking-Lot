from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


Box = Tuple[float, float, float, float, float, int]  # x1,y1,x2,y2,conf,cls


@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    conf: float = 0.25
    vehicle_class_ids: Tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck (COCO)


class Detector:
    """
    YOLOv8 detector wrapper.

    In starter mode, you can skip constructing this class and run with stub detections.
    """

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        self.config = config
        from ultralytics import YOLO  # imported lazily so API can boot without it if desired

        self.model = YOLO(config.model_path)

    def detect(self, frame) -> List[Box]:
        results = self.model.predict(frame, conf=self.config.conf, verbose=False)
        if not results:
            return []

        r0 = results[0]
        boxes = []
        for b in r0.boxes:
            cls_id = int(b.cls.item())
            if cls_id not in self.config.vehicle_class_ids:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            conf = float(b.conf.item())
            boxes.append((x1, y1, x2, y2, conf, cls_id))
        return boxes

