"""Render a dependency-free SVG energy diagnostic from diagnose_energy JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


def polyline(points: list[tuple[float, float]], css_class: str) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline class="{css_class}" points="{encoded}"/>'


def mask_runs(mask: np.ndarray, time_values: np.ndarray) -> list[tuple[float, float]]:
    result = []
    index = 0
    frame_s = float(time_values[1] - time_values[0]) if len(time_values) > 1 else 0.02
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and mask[end]:
            end += 1
        result.append((float(time_values[index]), float(time_values[end - 1] + frame_s)))
        index = end
    return result


def render_panel(
    index: int,
    start_s: float,
    end_s: float,
    time_values: np.ndarray,
    caller_db: np.ndarray,
    agent_db: np.ndarray,
    caller_official: np.ndarray,
    agent_official: np.ndarray,
    caller_extra: np.ndarray,
    caller_threshold: float,
    agent_threshold: float,
) -> str:
    left, right = 72.0, 1168.0
    top = 95.0 + index * 255.0
    bottom = top + 190.0
    plot_top = top + 22.0
    y_min, y_max = -80.0, -10.0

    def x(value: float) -> float:
        return left + (value - start_s) / (end_s - start_s) * (right - left)

    def y(value: float) -> float:
        clipped = min(y_max, max(y_min, value))
        return bottom - (clipped - y_min) / (y_max - y_min) * (bottom - plot_top)

    selected = (time_values >= start_s) & (time_values <= end_s)
    selected_indices = np.flatnonzero(selected)
    # One plotted point per 100 ms keeps the SVG small and readable.
    selected_indices = selected_indices[::5]
    caller_points = [(x(time_values[i]), y(caller_db[i])) for i in selected_indices]
    agent_points = [(x(time_values[i]), y(agent_db[i])) for i in selected_indices]

    parts = [f'<g aria-label="Energía de {start_s:.0f} a {end_s:.0f} segundos">']
    parts.append(f'<rect class="plot" x="{left}" y="{plot_top}" width="{right-left}" height="{bottom-plot_top}"/>')
    for value in (-80, -60, -40, -20):
        yy = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{yy}" x2="{right}" y2="{yy}"/>')
        parts.append(f'<text class="axis" x="{left-10}" y="{yy+4}" text-anchor="end">{value}</text>')
    tick = int(start_s // 10 * 10)
    if tick < start_s:
        tick += 10
    while tick <= end_s:
        xx = x(tick)
        parts.append(f'<line class="grid" x1="{xx}" y1="{plot_top}" x2="{xx}" y2="{bottom}"/>')
        parts.append(f'<text class="axis" x="{xx}" y="{bottom+18}" text-anchor="middle">{tick}s</text>')
        tick += 10

    extra_intervals = mask_runs(caller_extra, time_values)
    for interval_start, interval_end in extra_intervals:
        visible_start = max(start_s, interval_start)
        visible_end = min(end_s, interval_end)
        if visible_start < visible_end:
            parts.append(
                f'<rect class="extra" x="{x(visible_start):.2f}" y="{plot_top}" '
                f'width="{max(1.0, x(visible_end)-x(visible_start)):.2f}" height="{bottom-plot_top}"/>'
            )

    for css_class, mask, lane_y in (
        ("official-caller", caller_official, top),
        ("official-agent", agent_official, top + 9),
    ):
        for interval_start, interval_end in mask_runs(mask, time_values):
            visible_start = max(start_s, interval_start)
            visible_end = min(end_s, interval_end)
            if visible_start < visible_end:
                parts.append(
                    f'<rect class="{css_class}" x="{x(visible_start):.2f}" y="{lane_y}" '
                    f'width="{max(1.0, x(visible_end)-x(visible_start)):.2f}" height="6"/>'
                )

    parts.append(f'<line class="caller-threshold" x1="{left}" y1="{y(caller_threshold)}" x2="{right}" y2="{y(caller_threshold)}"/>')
    parts.append(f'<line class="agent-threshold" x1="{left}" y1="{y(agent_threshold)}" x2="{right}" y2="{y(agent_threshold)}"/>')
    parts.append(polyline(caller_points, "caller-line"))
    parts.append(polyline(agent_points, "agent-line"))
    parts.append(f'<text class="panel-label" x="{left}" y="{top-10}">{start_s:.0f}–{end_s:.0f} segundos</text>')
    parts.append(f'<text class="axis-title" x="20" y="{(plot_top+bottom)/2}" transform="rotate(-90 20 {(plot_top+bottom)/2})">Energía (dBFS)</text>')
    parts.append("</g>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnosis_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.diagnosis_json.read_text(encoding="utf-8"))
    energy = report["energy_dbfs"]
    time_values = np.asarray(energy["time_s"], dtype=float)
    caller_db = np.asarray(energy["caller"], dtype=float)
    agent_db = np.asarray(energy["agent"], dtype=float)
    caller_official = np.asarray(energy["caller_official"], dtype=bool)
    agent_official = np.asarray(energy["agent_official"], dtype=bool)
    caller_extra = np.asarray(energy["caller_extra"], dtype=bool)
    duration = float(report["duration_s"])
    boundaries = np.linspace(0.0, duration, 4)
    panels = [
        render_panel(
            index,
            boundaries[index],
            boundaries[index + 1],
            time_values,
            caller_db,
            agent_db,
            caller_official,
            agent_official,
            caller_extra,
            float(report["caller_threshold_dbfs"]),
            float(report["agent_threshold_dbfs"]),
        )
        for index in range(3)
    ]

    title = html.escape(f'{report["call_id"]} — energía por canal')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="870" viewBox="0 0 1200 870" role="img" aria-labelledby="title desc">
<title id="title">{title}</title>
<desc id="desc">Energía del caller y agente; las zonas moradas son detecciones adicionales del caller.</desc>
<style>
svg {{ background:#ffffff; color:#172033; font-family:system-ui,-apple-system,sans-serif; }}
.title {{ fill:#172033; font-size:24px; font-weight:600; }}
.subtitle,.axis,.axis-title,.legend {{ fill:#5f6b7a; font-size:12px; }}
.panel-label {{ fill:#172033; font-size:14px; font-weight:600; }}
.plot {{ fill:#f8fafc; stroke:#d9e0e8; }}
.grid {{ stroke:#d9e0e8; stroke-width:1; }}
.caller-line {{ fill:none; stroke:#2563eb; stroke-width:1.7; }}
.agent-line {{ fill:none; stroke:#f59e0b; stroke-width:1.5; opacity:.85; }}
.caller-threshold {{ stroke:#2563eb; stroke-width:1; stroke-dasharray:5 4; opacity:.8; }}
.agent-threshold {{ stroke:#f59e0b; stroke-width:1; stroke-dasharray:5 4; opacity:.8; }}
.official-caller {{ fill:#16a34a; }}
.official-agent {{ fill:#0891b2; }}
.extra {{ fill:#8b5cf6; opacity:.18; }}
.swatch-caller {{ fill:#2563eb; }} .swatch-agent {{ fill:#f59e0b; }} .swatch-extra {{ fill:#8b5cf6; opacity:.5; }}
@media (prefers-color-scheme:dark) {{ svg {{ background:#111827; color:#e5e7eb; }} .title,.panel-label {{ fill:#e5e7eb; }} .subtitle,.axis,.axis-title,.legend {{ fill:#a9b2c1; }} .plot {{ fill:#182235; stroke:#344156; }} .grid {{ stroke:#344156; }} }}
</style>
<text class="title" x="72" y="36">{title}</text>
<text class="subtitle" x="72" y="60">Umbral caller {report["caller_threshold_dbfs"]:.1f} dBFS · voz adicional {report["caller_extra_s"]:.2f}s · morado = Energy VAD activo sin turno oficial</text>
<g class="legend" transform="translate(690 35)">
  <rect class="swatch-caller" x="0" y="-9" width="18" height="4"/><text x="24" y="0">Caller</text>
  <rect class="swatch-agent" x="100" y="-9" width="18" height="4"/><text x="124" y="0">Agente</text>
  <rect class="swatch-extra" x="210" y="-14" width="18" height="14"/><text x="234" y="0">Detección adicional</text>
  <rect class="official-caller" x="0" y="12" width="18" height="6"/><text x="24" y="20">Turno oficial caller</text>
  <rect class="official-agent" x="180" y="12" width="18" height="6"/><text x="204" y="20">Turno oficial agente</text>
</g>
{''.join(panels)}
</svg>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
