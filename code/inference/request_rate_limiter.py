"""Process-shared fixed-window request limiter for hosted evaluation runs.

The limiter is opt-in through environment variables so ordinary inference is
unchanged.  Reservations are persisted before a request is sent; a crashed or
failed request therefore consumes one conservative slot instead of risking a
provider quota overrun.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time
from typing import Callable


STATE_ENV = "PROACTIVE_AGENT_RATE_LIMIT_STATE"
MAX_ENV = "PROACTIVE_AGENT_RATE_LIMIT_MAX_REQUESTS"
WINDOW_ENV = "PROACTIVE_AGENT_RATE_LIMIT_WINDOW_SECONDS"
INTERVAL_ENV = "PROACTIVE_AGENT_RATE_LIMIT_MIN_INTERVAL_SECONDS"


class SharedRequestRateLimiter:
    def __init__(
        self,
        *,
        state_path: Path,
        max_requests: int,
        window_seconds: float,
        min_interval_seconds: float,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self.state_path = state_path
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_interval_seconds = min_interval_seconds
        self.clock = clock
        self.sleeper = sleeper

    @classmethod
    def from_environment(cls) -> "SharedRequestRateLimiter | None":
        values = {
            STATE_ENV: os.environ.get(STATE_ENV),
            MAX_ENV: os.environ.get(MAX_ENV),
            WINDOW_ENV: os.environ.get(WINDOW_ENV),
            INTERVAL_ENV: os.environ.get(INTERVAL_ENV),
        }
        configured = [key for key, value in values.items() if value not in (None, "")]
        if not configured:
            return None
        missing = [key for key, value in values.items() if value in (None, "")]
        if missing:
            raise ValueError("Incomplete request-rate-limit configuration: " + ", ".join(missing))
        return cls(
            state_path=Path(values[STATE_ENV]).expanduser().resolve(),
            max_requests=int(values[MAX_ENV]),
            window_seconds=float(values[WINDOW_ENV]),
            min_interval_seconds=float(values[INTERVAL_ENV]),
        )

    @property
    def public_config(self) -> dict[str, object]:
        return {
            "implementation": "proseekbench.shared-fixed-window.v1",
            "state_path": str(self.state_path),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "min_interval_seconds": self.min_interval_seconds,
            "reservation_policy": "reserve_before_transport",
        }

    def _load(self, stream) -> dict[str, float | int | None]:
        stream.seek(0)
        raw = stream.read().strip()
        if not raw:
            return {"window_started_at": None, "count": 0, "last_request_at": None}
        value = json.loads(raw)
        return {
            "window_started_at": value.get("window_started_at"),
            "count": int(value.get("count", 0)),
            "last_request_at": value.get("last_request_at"),
        }

    def _save(self, stream, state: dict[str, float | int | None]) -> None:
        stream.seek(0)
        stream.truncate()
        json.dump(state, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    def acquire(self) -> dict[str, object]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        waited = 0.0
        while True:
            wait_for = 0.0
            with self.state_path.open("a+", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                state = self._load(stream)
                now = self.clock()
                started = state["window_started_at"]
                if started is None or now >= float(started) + self.window_seconds:
                    state = {"window_started_at": now, "count": 0, "last_request_at": None}
                    started = now
                if int(state["count"]) >= self.max_requests:
                    wait_for = max(0.0, float(started) + self.window_seconds - now)
                elif state["last_request_at"] is not None:
                    wait_for = max(
                        0.0,
                        float(state["last_request_at"]) + self.min_interval_seconds - now,
                    )
                if wait_for <= 0:
                    state["count"] = int(state["count"]) + 1
                    state["last_request_at"] = now
                    self._save(stream, state)
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    return {
                        "window_started_at": state["window_started_at"],
                        "window_count": state["count"],
                        "window_remaining": self.max_requests - int(state["count"]),
                        "waited_seconds": round(waited, 3),
                    }
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            self.sleeper(wait_for)
            waited += wait_for
