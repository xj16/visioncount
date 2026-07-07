"""Tests for the crossing event store (ring buffer, SQLite, CSV export)."""

from __future__ import annotations

from visioncount.counter import CrossEvent
from visioncount.events import CSV_COLUMNS, EventStore, StoredEvent


def _ev(seq: int) -> CrossEvent:
    return CrossEvent(line="door", track_id=seq, direction="in", point=(seq, seq * 2))


def test_record_enriches_with_seq_and_iso():
    store = EventStore(clock=lambda: 1_700_000_000.0)
    stored = store.record(_ev(1))
    assert isinstance(stored, StoredEvent)
    assert stored.seq == 0
    assert stored.line == "door"
    assert stored.x == 1 and stored.y == 2
    assert stored.iso.endswith("Z") and "T" in stored.iso


def test_seq_is_monotonic_and_count_tracks_all():
    store = EventStore()
    for i in range(5):
        store.record(_ev(i))
    assert store.count() == 5
    recent = store.recent(10)
    # Most-recent first.
    assert [e["seq"] for e in recent] == [4, 3, 2, 1, 0]


def test_ring_buffer_bounds_memory_but_count_is_total():
    store = EventStore(maxlen=3)
    for i in range(10):
        store.record(_ev(i))
    recent = store.recent(100)
    assert len(recent) == 3  # ring holds only the last 3
    assert [e["seq"] for e in recent] == [9, 8, 7]
    assert store.count() == 10  # but total is preserved


def test_csv_export_header_and_rows():
    store = EventStore()
    store.record(_ev(0))
    store.record(_ev(1))
    rows = list(store.iter_csv())
    assert rows[0].strip() == ",".join(CSV_COLUMNS)
    assert len(rows) == 3  # header + 2 events
    assert rows[1].startswith("0,")
    assert "door" in rows[1]


def test_sqlite_persistence_and_resume(tmp_path):
    db = str(tmp_path / "events.sqlite")
    store = EventStore(db_path=db)
    for i in range(4):
        store.record(_ev(i))
    store.close()

    # Reopen: seq must resume so ids never collide across restarts.
    store2 = EventStore(db_path=db)
    stored = store2.record(_ev(99))
    assert stored.seq == 4
    # CSV export reads from SQLite and includes all 5 rows.
    rows = list(store2.iter_csv())
    assert len(rows) == 6  # header + 5
    store2.close()


def test_sqlite_export_orders_by_seq(tmp_path):
    db = str(tmp_path / "e.sqlite")
    store = EventStore(db_path=db)
    for i in range(3):
        store.record(_ev(i))
    seqs = [line.split(",")[0] for line in list(store.iter_csv())[1:]]
    assert seqs == ["0", "1", "2"]
    store.close()


def test_invalid_maxlen_rejected():
    import pytest

    with pytest.raises(ValueError):
        EventStore(maxlen=0)
