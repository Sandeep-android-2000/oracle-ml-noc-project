"""1-minute APScheduler loop that drives the live pull pipeline.

State (var, started, last_ts) is stored on the singleton instance so the
server can expose `/api/live/status`, `/api/live/start`, `/api/live/stop`.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from motor.motor_asyncio import AsyncIOMotorDatabase

from .pipeline import online_retrain, pull_one_var, RETRAIN_EVERY_N_TICKS

logger = logging.getLogger("ml.scheduler")

INTERVAL_SECONDS = 60


class LiveScheduler:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.var: int = 0
        self.last_tick_ts: Optional[str] = None
        self.last_tick_result: Optional[Dict[str, Any]] = None
        self.last_retrain: Optional[Dict[str, Any]] = None
        self.running: bool = False
        self._lock = asyncio.Lock()

    async def _tick(self):
        async with self._lock:
            try:
                result = await pull_one_var(self.db, self.var)
                self.last_tick_result = result
                self.last_tick_ts = datetime.now(timezone.utc).isoformat()
                logger.info("tick var=%d pulled=%d", self.var, result.get("pulled", 0))
                self.var += 1

                if self.var % RETRAIN_EVERY_N_TICKS == 0:
                    logger.info("online retrain at var=%d", self.var)
                    self.last_retrain = await online_retrain(self.db)
            except Exception as exc:  # noqa: BLE001
                logger.exception("tick failed: %s", exc)

    async def start(self, interval_seconds: int = INTERVAL_SECONDS,
                    reset_var: bool = False) -> Dict[str, Any]:
        if self.running and self.scheduler:
            return {"status": "already_running", "var": self.var,
                    "interval_seconds": interval_seconds}
        if reset_var:
            self.var = 0
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.scheduler.add_job(
            self._tick, "interval", seconds=interval_seconds,
            id="live_pull", max_instances=1, coalesce=True,
            next_run_time=datetime.now(timezone.utc),  # fire immediately
        )
        self.scheduler.start()
        self.running = True
        return {"status": "started", "var": self.var,
                "interval_seconds": interval_seconds}

    async def stop(self) -> Dict[str, Any]:
        if self.scheduler and self.running:
            self.scheduler.shutdown(wait=False)
            self.running = False
            return {"status": "stopped", "var": self.var}
        return {"status": "not_running"}

    async def tick_once(self) -> Dict[str, Any]:
        await self._tick()
        return self.last_tick_result or {"pulled": 0}

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "current_var": self.var,
            "last_tick_ts": self.last_tick_ts,
            "last_tick": self.last_tick_result,
            "last_retrain": self.last_retrain,
            "interval_seconds": INTERVAL_SECONDS,
            "retrain_every_n_ticks": RETRAIN_EVERY_N_TICKS,
        }


LIVE: Optional[LiveScheduler] = None


def get_scheduler(db: AsyncIOMotorDatabase) -> LiveScheduler:
    global LIVE
    if LIVE is None:
        LIVE = LiveScheduler(db)
    return LIVE
