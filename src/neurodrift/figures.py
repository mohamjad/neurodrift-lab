"""Dependency-light SVG figures for evidence reports."""

from __future__ import annotations

from pathlib import Path

from neurodrift.evidence import EvidenceSummary


def write_evidence_figures(summary: EvidenceSummary, output_dir: Path) -> tuple[Path, ...]:
    """Write stable SVG figures for a NeuroDrift evidence summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "split-rate.svg",
        output_dir / "mse-vs-meaning.svg",
    )
    paths[0].write_text(render_split_rate(summary), encoding="utf-8")
    paths[1].write_text(render_mse_vs_meaning(summary), encoding="utf-8")
    return paths


def render_split_rate(summary: EvidenceSummary) -> str:
    """Render split and over-alignment rates as a compact SVG bar figure."""

    stats = summary.statistics
    split = float(stats["split_rate"]["mean"])
    overalign = float(stats["overalignment_rate"]["mean"])
    bars = (("alignment/meaning split", split, "#d8d8d8"), ("over-alignment", overalign, "#8f8f8f"))
    width = 760
    height = 260
    left = 210
    max_width = 430
    rows = []
    for idx, (label, value, color) in enumerate(bars):
        y = 78 + idx * 76
        bar_width = max(1, int(max_width * max(0.0, min(value, 1.0))))
        rows.append(
            f'<text x="32" y="{y + 23}" class="label">{label}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_width}" height="36" fill="{color}" />'
            f'<text x="{left + bar_width + 14}" y="{y + 24}" class="value">{value:.2f}</text>'
        )
    return _svg_shell(
        width,
        height,
        "neurodrift evidence rates",
        "\n".join(rows),
    )


def render_mse_vs_meaning(summary: EvidenceSummary) -> str:
    """Render best-MSE gain against best meaning distance."""

    width = 760
    height = 420
    left = 76
    top = 56
    plot_width = 600
    plot_height = 280
    gains = [run.best_mse_alignment_gain for run in summary.runs]
    distances = [run.best_meaning_distance for run in summary.runs]
    min_gain = min(gains)
    max_gain = max(gains)
    min_dist = min(distances)
    max_dist = max(distances)

    def x_pos(value: float) -> float:
        span = max(max_gain - min_gain, 1e-8)
        return left + ((value - min_gain) / span) * plot_width

    def y_pos(value: float) -> float:
        span = max(max_dist - min_dist, 1e-8)
        return top + plot_height - ((value - min_dist) / span) * plot_height

    points = []
    for run in summary.runs:
        fill = "#ffffff" if run.has_alignment_meaning_split else "#9a9a9a"
        stroke = "#111111" if run.overaligns else "#6f6f6f"
        points.append(
            f'<circle cx="{x_pos(run.best_mse_alignment_gain):.2f}" '
            f'cy="{y_pos(run.best_meaning_distance):.2f}" r="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
        )
    body = "\n".join(
        [
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
            'fill="none" stroke="#777" stroke-width="1" />',
            *points,
            f'<text x="{left}" y="{top + plot_height + 44}" class="axis">'
            "best MSE alignment gain</text>",
            (
                f'<text x="{left - 44}" y="{top + 12}" class="axis" '
                'transform="rotate(-90 32 68)">best meaning distance</text>'
            ),
            f'<text x="{left}" y="{top + plot_height + 22}" class="tick">{min_gain:.2f}</text>',
            f'<text x="{left + plot_width - 36}" y="{top + plot_height + 22}" '
            f'class="tick">{max_gain:.2f}</text>',
            f'<text x="{left - 52}" y="{top + plot_height}" class="tick">{min_dist:.2f}</text>',
            f'<text x="{left - 52}" y="{top + 4}" class="tick">{max_dist:.2f}</text>',
        ]
    )
    return _svg_shell(width, height, "mse gain vs meaning distance", body)


def _svg_shell(width: int, height: int, title: str, body: str) -> str:
    view_box = f"0 0 {width} {height}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="{view_box}">\n'
        f"""
  <rect width="{width}" height="{height}" fill="#050505" />
  <style>
    text {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #e8e8e8; }}
    .title {{ font-size: 22px; font-weight: 600; }}
    .label {{ font-size: 15px; }}
    .value {{ font-size: 15px; fill: #f2f2f2; }}
    .axis {{ font-size: 13px; fill: #c9c9c9; }}
    .tick {{ font-size: 11px; fill: #b4b4b4; }}
  </style>
  <text x="32" y="38" class="title">{title}</text>
  {body}
</svg>
"""
    )
