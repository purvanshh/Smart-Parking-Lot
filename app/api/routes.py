from __future__ import annotations

from fastapi import APIRouter

from app.occupancy.engine import OccupancyEngine


def build_router(occupancy_engine: OccupancyEngine) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health():
        return {"status": "ok", "last_update_ts": occupancy_engine.last_update_ts}

    @router.get("/slots")
    def get_slots():
        return occupancy_engine.slot_status

    @router.get("/summary")
    def get_summary():
        return occupancy_engine.summary()

    return router

