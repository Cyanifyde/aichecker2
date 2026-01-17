from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    level = (level or os.environ.get("AICHECKER_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=LOG_FORMAT, datefmt=LOG_DATEFMT)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


@dataclass
class Progress:
    total: int
    log_every: int = 50
    label: str = "progress"

    start_time: float = 0.0
    seen: int = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.seen = 0

    def update(self, n: int = 1) -> None:
        self.seen += int(n)

    def should_log(self) -> bool:
        if self.total <= 0:
            return False
        if self.seen <= 0:
            return False
        if self.seen == self.total:
            return True
        return (self.seen % max(1, self.log_every)) == 0

    def stats(self) -> tuple[str, float, float]:
        elapsed = max(0.0, time.perf_counter() - self.start_time)
        rate = (self.seen / elapsed) if elapsed > 0 else 0.0
        remaining = (self.total - self.seen)
        eta = (remaining / rate) if rate > 0 else float("inf")
        msg = f"{self.label}: {self.seen}/{self.total} ({rate:.2f}/s, eta {format_duration(eta) if eta != float('inf') else '?:??'})"
        return msg, elapsed, eta

