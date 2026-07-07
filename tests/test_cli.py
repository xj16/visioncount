"""Tests for the command-line entrypoint."""

from __future__ import annotations

import json
import sqlite3

from visioncount.cli import _parse_line, _parse_zone, main


def test_parse_line_ok():
    name, start, end = _parse_line("door:0,240,640,240")
    assert name == "door"
    assert start == (0, 240)
    assert end == (640, 240)


def test_parse_line_rejects_garbage():
    import argparse

    import pytest

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_line("nope")


def test_parse_zone_ok():
    name, pts = _parse_zone("roi:0,0;100,0;100,100;0,100")
    assert name == "roi"
    assert pts == [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_parse_zone_requires_three_points():
    import argparse

    import pytest

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_zone("roi:0,0;100,0")


def test_main_runs_synthetic_headless_json(capsys):
    rc = main(
        [
            "--source", "synthetic",
            "--width", "320", "--height", "240",
            "--frames", "120", "--no-window", "--json",
            "--line", "center:160,0,160,240",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert "lines" in payload and "zones" in payload
    assert "center" in payload["lines"]


def test_main_logs_to_sqlite(tmp_path, capsys):
    db = str(tmp_path / "run.sqlite")
    rc = main(
        [
            "--source", "synthetic",
            "--width", "320", "--height", "240",
            "--frames", "160", "--no-window",
            "--min-area", "200",
            "--line", "center:160,0,160,240",
            "--zone", "roi:80,60;240,60;240,180;80,180",
            "--db", db,
        ]
    )
    assert rc == 0
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM crossings").fetchone()[0]
    assert n >= 1
    # Every row has a well-formed direction.
    dirs = {d for (d,) in conn.execute("SELECT DISTINCT direction FROM crossings")}
    assert dirs <= {"in", "out"}
    conn.close()
