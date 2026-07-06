"""Matplotlib analytics reports.

Turns the running counts of a :class:`~visioncount.pipeline.Pipeline` into a
static PNG report: a bar chart of in/out per line and a cumulative
crossings-over-time line chart. Uses the non-interactive ``Agg`` backend so it
works headless (servers, CI, a Raspberry Pi with no display).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402


def _series_to_xy(series: Sequence[Tuple[float, int]]) -> Tuple[List[float], List[int]]:
    if not series:
        return [], []
    t0 = series[0][0]
    xs = [round(t - t0, 2) for t, _ in series]
    ys = [v for _, v in series]
    return xs, ys


def render_report(
    totals: Dict[str, Dict[str, int]],
    series: Sequence[Tuple[float, int]],
    path: str,
    title: str = "VisionCount report",
) -> str:
    """Render a two-panel PNG report to ``path`` and return the path.

    Parameters
    ----------
    totals:
        Mapping of line name -> ``{"in", "out", "total", "net"}`` (from
        ``Pipeline.totals()``).
    series:
        Sequence of ``(timestamp, cumulative_total)`` samples.
    path:
        Output PNG path.
    title:
        Figure title.
    """
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---- Left: per-line in/out bar chart --------------------------------
    names = list(totals.keys())
    if names:
        ins = [totals[n]["in"] for n in names]
        outs = [totals[n]["out"] for n in names]
        x = range(len(names))
        width = 0.38
        ax_bar.bar([i - width / 2 for i in x], ins, width, label="in", color="#1f7a8c")
        ax_bar.bar([i + width / 2 for i in x], outs, width, label="out", color="#e07a3f")
        ax_bar.set_xticks(list(x))
        ax_bar.set_xticklabels(names)
        ax_bar.legend()
    else:
        ax_bar.text(0.5, 0.5, "no lines", ha="center", va="center")
    ax_bar.set_title("Crossings by line")
    ax_bar.set_ylabel("count")
    ax_bar.grid(axis="y", alpha=0.3)

    # ---- Right: cumulative crossings over time --------------------------
    xs, ys = _series_to_xy(series)
    if xs:
        ax_line.plot(xs, ys, color="#0f2f4a", linewidth=2)
        ax_line.fill_between(xs, ys, color="#0f2f4a", alpha=0.08)
    else:
        ax_line.text(0.5, 0.5, "no samples", ha="center", va="center")
    ax_line.set_title("Total crossings over time")
    ax_line.set_xlabel("seconds")
    ax_line.set_ylabel("cumulative total")
    ax_line.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
