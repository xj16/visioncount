"""Crossing-event persistence, export and outbound notification.

The counting pipeline emits a :class:`~visioncount.counter.CrossEvent` every time
a tracked object crosses a line. On its own that event is ephemeral -- it lives
only long enough to bump an in-memory counter. For an *analytics* product that is
not enough: you want the individual crossings, with timestamps, so you can export
them, chart them, or forward them somewhere.

:class:`EventStore` closes that gap. Every crossing is appended to:

* an in-memory ring buffer (always on, cheap, bounded), and
* an optional SQLite database (``VISIONCOUNT_DB`` / ``--db``), so a run survives
  a restart and can be queried with plain SQL, and
* an optional outbound webhook (``VISIONCOUNT_WEBHOOK``), fired best-effort on a
  daemon thread so a slow endpoint never stalls the video loop.

Nothing here imports OpenCV or NumPy, so it stays trivially unit-testable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Deque, Dict, List, Optional
from collections import deque
from urllib import request as _urlrequest

from .counter import CrossEvent

# Public CSV header, also used by the streaming exporter and the tests.
CSV_COLUMNS = ("seq", "ts", "iso", "line", "direction", "track_id", "x", "y")


@dataclass(frozen=True)
class StoredEvent:
    """A crossing enriched with a monotonically increasing seq and a timestamp."""

    seq: int
    ts: float  # unix epoch seconds
    line: str
    direction: str  # "in" | "out"
    track_id: int
    x: int
    y: int

    @property
    def iso(self) -> str:
        """UTC ISO-8601 timestamp, e.g. ``2026-07-07T04:38:00Z``."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ts))

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["iso"] = self.iso
        return d

    def csv_row(self) -> str:
        return ",".join(
            str(v)
            for v in (
                self.seq,
                f"{self.ts:.3f}",
                self.iso,
                self.line,
                self.direction,
                self.track_id,
                self.x,
                self.y,
            )
        )


class EventStore:
    """Thread-safe store of crossing events.

    Parameters
    ----------
    maxlen:
        Size of the in-memory ring buffer (recent events kept for ``/api/events``).
    db_path:
        Optional SQLite path. When set, every event is also inserted there. Use
        ``":memory:"`` for an ephemeral database (handy in tests).
    webhook_url:
        Optional URL to ``POST`` each event to as JSON. Fired on a background
        daemon thread; failures are counted, never raised.
    clock:
        Injectable time source (defaults to :func:`time.time`) so tests can pin
        deterministic timestamps.
    """

    def __init__(
        self,
        maxlen: int = 500,
        db_path: Optional[str] = None,
        webhook_url: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        self._lock = threading.Lock()
        self._ring: Deque[StoredEvent] = deque(maxlen=maxlen)
        self._seq = 0
        self._clock = clock
        self.webhook_url = webhook_url
        self.webhook_failures = 0
        self._db: Optional[sqlite3.Connection] = None
        self.db_path = db_path
        if db_path:
            # check_same_thread=False: the worker thread writes, request threads read.
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS crossings (
                    seq       INTEGER PRIMARY KEY,
                    ts        REAL    NOT NULL,
                    iso       TEXT    NOT NULL,
                    line      TEXT    NOT NULL,
                    direction TEXT    NOT NULL,
                    track_id  INTEGER NOT NULL,
                    x         INTEGER NOT NULL,
                    y         INTEGER NOT NULL
                )
                """
            )
            self._db.commit()
            # Resume the sequence counter after a restart so seqs never collide.
            row = self._db.execute("SELECT MAX(seq) FROM crossings").fetchone()
            if row and row[0] is not None:
                self._seq = int(row[0]) + 1

    # ---- ingest ----------------------------------------------------------

    def record(self, event: CrossEvent) -> StoredEvent:
        """Persist one :class:`CrossEvent`; return the enriched stored form."""
        with self._lock:
            stored = StoredEvent(
                seq=self._seq,
                ts=self._clock(),
                line=event.line,
                direction=event.direction,
                track_id=event.track_id,
                x=int(event.point[0]),
                y=int(event.point[1]),
            )
            self._seq += 1
            self._ring.append(stored)
            if self._db is not None:
                self._db.execute(
                    "INSERT INTO crossings VALUES (?,?,?,?,?,?,?,?)",
                    (
                        stored.seq,
                        stored.ts,
                        stored.iso,
                        stored.line,
                        stored.direction,
                        stored.track_id,
                        stored.x,
                        stored.y,
                    ),
                )
                self._db.commit()
        if self.webhook_url:
            self._fire_webhook(stored)
        return stored

    def record_many(self, events: List[CrossEvent]) -> List[StoredEvent]:
        return [self.record(e) for e in events]

    # ---- query -----------------------------------------------------------

    def recent(self, limit: int = 100) -> List[Dict[str, object]]:
        """Most-recent events first, capped at ``limit`` (JSON-ready dicts)."""
        with self._lock:
            items = list(self._ring)[-limit:]
        items.reverse()
        return [e.as_dict() for e in items]

    def count(self) -> int:
        """Total events ever recorded (across the whole run)."""
        with self._lock:
            return self._seq

    def iter_csv(self):
        """Yield CSV lines for *all* crossings (SQLite if present, else the ring).

        A generator so the Flask layer can stream a large export without building
        the whole file in memory.
        """
        yield ",".join(CSV_COLUMNS) + "\n"
        if self._db is not None:
            # A separate read cursor keeps the stream independent of writes.
            cur = self._db.cursor()
            for row in cur.execute(
                "SELECT seq, ts, iso, line, direction, track_id, x, y "
                "FROM crossings ORDER BY seq"
            ):
                seq, ts, iso, line, direction, track_id, x, y = row
                yield f"{seq},{ts:.3f},{iso},{line},{direction},{track_id},{x},{y}\n"
        else:
            with self._lock:
                snapshot = list(self._ring)
            for e in snapshot:
                yield e.csv_row() + "\n"

    # ---- webhook ---------------------------------------------------------

    def _fire_webhook(self, stored: StoredEvent) -> None:
        payload = json.dumps(stored.as_dict()).encode("utf-8")

        def _post() -> None:
            try:
                req = _urlrequest.Request(
                    self.webhook_url,  # type: ignore[arg-type]
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                _urlrequest.urlopen(req, timeout=3.0).close()
            except Exception:  # noqa: BLE001 - best-effort; never crash the loop
                with self._lock:
                    self.webhook_failures += 1

        threading.Thread(target=_post, daemon=True).start()

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
