from __future__ import annotations

from typing import Callable, Dict

from fastapi import APIRouter

from app.analytics.engine import AnalyticsEngine
from app.occupancy.engine import OccupancyEngine


def build_router(
    occupancy_engine: OccupancyEngine,
    analytics_engine: AnalyticsEngine,
    health_provider: Callable[[], Dict],
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health():
        return health_provider()

    @router.get("/slots")
    def get_slots():
        return occupancy_engine.slot_status

    @router.get("/summary")
    def get_summary():
        return occupancy_engine.summary()

    @router.get("/analytics")
    def get_analytics():
        s = analytics_engine.snapshot()
        return {
            "total_vehicles": s.total_vehicles_served,
            "avg_duration_s": s.average_parking_duration_s,
            "utilization": s.slot_utilization_rate,
            "busiest_slots": s.busiest_slots,
            "least_used_slots": s.least_used_slots,
            "avg_duration_per_slot_s": s.average_duration_per_slot_s,
        }

    @router.get("/sessions")
    def get_sessions(limit: int = 50):
        sessions = analytics_engine.recent_sessions(limit=limit)
        # ParkingSession is a dataclass -> __dict__ is safe for JSON-ish output.
        return [s.__dict__ for s in sessions]

    @router.get("/alerts")
    def get_alerts():
        return [a.__dict__ for a in analytics_engine.alerts()]

    return router

