"""Generate a static coverage badge SVG from coverage.json.

Runs after ``pytest --cov --cov-report=json:coverage.json`` and writes
``docs/coverage.svg`` — a self-contained shields.io-style badge with no network
dependency, so the README badge works even offline / on a fork.

    python examples/make_coverage_badge.py
"""

from __future__ import annotations

import json
import os
import sys

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "coverage.svg")
COV_JSON = os.path.join(os.path.dirname(__file__), "..", "coverage.json")


def color_for(pct: float) -> str:
    if pct >= 90:
        return "#4c1"      # brightgreen
    if pct >= 80:
        return "#97ca00"   # green
    if pct >= 70:
        return "#a4a61d"   # yellowgreen
    if pct >= 60:
        return "#dfb317"   # yellow
    return "#e05d44"       # red


def render(pct: float) -> str:
    label, value = "coverage", f"{pct:.0f}%"
    color = color_for(pct)
    # Approximate glyph widths at 11px for the two segments.
    lw, vw = 62, 44
    total = lw + vw
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" role="img" aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{lw}" height="20" fill="#555"/>
    <rect x="{lw}" width="{vw}" height="20" fill="{color}"/>
    <rect width="{total}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{lw / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{lw / 2:.0f}" y="14">{label}</text>
    <text x="{lw + vw / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{lw + vw / 2:.0f}" y="14">{value}</text>
  </g>
</svg>
"""


def main() -> int:
    if not os.path.exists(COV_JSON):
        print("coverage.json not found; run pytest with --cov-report=json first",
              file=sys.stderr)
        return 1
    data = json.load(open(COV_JSON))
    pct = float(data["totals"]["percent_covered"])
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(render(pct))
    print(f"Wrote {OUT} ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
