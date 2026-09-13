"""Generate an HTML report comparing official and energy-VAD timelines.

The report automatically includes every misclassified validation call found in
the predictions CSV, so it can be regenerated after each new experiment.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from ..feature_extraction import load_turns
from ..vad import VadConfig, detect_turns_from_wav


def frame_mask(turns: list[dict], channel: int, frame_count: int, frame_s: float) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    for turn in turns:
        if turn["channel"] != channel:
            continue
        start = max(0, int(math.floor(turn["start"] / frame_s)))
        end = min(frame_count, int(math.ceil(turn["end"] / frame_s)))
        mask[start:end] = True
    return mask


def runs(mask: np.ndarray, frame_s: float) -> list[tuple[float, float]]:
    intervals = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and mask[end]:
            end += 1
        intervals.append((index * frame_s, end * frame_s))
        index = end
    return intervals


def lane_segments(intervals: list[tuple[float, float]], duration: float, css_class: str) -> str:
    parts = []
    for start, end in intervals:
        left = 100.0 * start / duration
        width = max(0.08, 100.0 * (end - start) / duration)
        label = f"{start:.2f}–{end:.2f} s"
        parts.append(
            f'<span class="segment {css_class}" style="left:{left:.5f}%;width:{width:.5f}%" '
            f'aria-label="{html.escape(label)}"></span>'
        )
    return "".join(parts)


def lane(label: str, intervals: list[tuple[float, float]], duration: float, css_class: str) -> str:
    return (
        '<div class="lane">'
        f'<div class="lane-label">{html.escape(label)}</div>'
        '<div class="track">'
        f'{lane_segments(intervals, duration, css_class)}'
        '</div></div>'
    )


def mask_metrics(reference: np.ndarray, predicted: np.ndarray, frame_s: float) -> dict:
    tp = int(np.sum(reference & predicted))
    fp = int(np.sum(~reference & predicted))
    fn = int(np.sum(reference & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "missing_s": fn * frame_s,
        "extra_s": fp * frame_s,
    }


def time_axis(duration: float) -> str:
    step = 30 if duration <= 210 else 60
    ticks = list(range(0, int(duration) + 1, step))
    if not ticks or duration - ticks[-1] > 10:
        ticks.append(int(round(duration)))
    parts = ['<div class="axis" aria-label="Tiempo de la llamada en segundos">']
    for value in ticks:
        left = min(100.0, 100.0 * value / duration)
        parts.append(
            f'<span class="tick" style="left:{left:.5f}%"><span>{value}s</span></span>'
        )
    parts.append("</div>")
    return "".join(parts)


def build_panel(
    row: dict,
    audio_dir: Path,
    turns_dir: Path,
    output_dir: Path,
    config: VadConfig,
) -> tuple[str, dict]:
    call_id = row["anon_id"]
    audio_path = audio_dir / f"{call_id}.wav"
    predicted_turns, diagnostics = detect_turns_from_wav(audio_path, config)
    official_turns = load_turns(turns_dir / f"{call_id}.json")
    duration = float(diagnostics["duration_s"])
    frame_s = config.frame_ms / 1000.0
    frame_count = int(math.ceil(duration / frame_s))

    masks = {}
    channel_metrics = {}
    for channel in (0, 1):
        official = frame_mask(official_turns, channel, frame_count, frame_s)
        predicted = frame_mask(predicted_turns, channel, frame_count, frame_s)
        masks[(channel, "official")] = official
        masks[(channel, "predicted")] = predicted
        masks[(channel, "missing")] = official & ~predicted
        masks[(channel, "extra")] = predicted & ~official
        channel_metrics[channel] = mask_metrics(official, predicted, frame_s)

    audio_relative = quote(os.path.relpath(audio_path, output_dir))
    probability = float(row["probability_synthetic"])
    truth = "Sintética" if row["label"] == "synthetic" else "Humana"
    prediction = "Sintética" if row["prediction"] == "synthetic" else "Humana"
    panel = [
        f'<section class="call-panel" id="panel-{html.escape(call_id)}" hidden>',
        '<div class="summary">',
        f'<div><strong>Real:</strong> {truth}</div>',
        f'<div><strong>Predicción:</strong> {prediction}</div>',
        f'<div><strong>Prob. sintética:</strong> {probability:.1%}</div>',
        f'<div><strong>Duración:</strong> {duration:.1f} s</div>',
        "</div>",
        f'<audio controls preload="metadata" src="{audio_relative}">Audio de {html.escape(call_id)}</audio>',
        '<div class="legend" aria-label="Leyenda">'
        '<span><i class="official"></i>Turnos oficiales</span>'
        '<span><i class="detected"></i>Energy VAD</span>'
        '<span><i class="missing"></i>Voz perdida</span>'
        '<span><i class="extra"></i>Voz adicional</span>'
        '</div>',
        '<h2>Caller — canal 0</h2>',
        time_axis(duration),
        lane("Oficial", runs(masks[(0, "official")], frame_s), duration, "official"),
        lane("Energy VAD", runs(masks[(0, "predicted")], frame_s), duration, "detected"),
        lane("Diferencias", runs(masks[(0, "missing")], frame_s), duration, "missing")[:-12]
        + lane_segments(runs(masks[(0, "extra")], frame_s), duration, "extra")
        + "</div></div>",
        f'<p class="metric">F1 {channel_metrics[0]["f1"]:.1%} · '
        f'voz perdida {channel_metrics[0]["missing_s"]:.2f}s · '
        f'voz adicional {channel_metrics[0]["extra_s"]:.2f}s</p>',
        '<h2>Agente — canal 1</h2>',
        time_axis(duration),
        lane("Oficial", runs(masks[(1, "official")], frame_s), duration, "official"),
        lane("Energy VAD", runs(masks[(1, "predicted")], frame_s), duration, "detected"),
        lane("Diferencias", runs(masks[(1, "missing")], frame_s), duration, "missing")[:-12]
        + lane_segments(runs(masks[(1, "extra")], frame_s), duration, "extra")
        + "</div></div>",
        f'<p class="metric">F1 {channel_metrics[1]["f1"]:.1%} · '
        f'voz perdida {channel_metrics[1]["missing_s"]:.2f}s · '
        f'voz adicional {channel_metrics[1]["extra_s"]:.2f}s</p>',
        "</section>",
    ]
    return "".join(panel), {
        "call_id": call_id,
        "truth": truth,
        "prediction": prediction,
        "probability": probability,
        "caller_f1": channel_metrics[0]["f1"],
        "agent_f1": channel_metrics[1]["f1"],
    }


def build_html(panels: list[str], summaries: list[dict]) -> str:
    buttons = "".join(
        f'<button type="button" class="call-button" data-call="{html.escape(item["call_id"])}">'
        f'{html.escape(item["call_id"].replace("call_", ""))}</button>'
        for item in summaries
    )
    summary_json = json.dumps(summaries, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Errores de clasificación — Energy VAD</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f8fb; --text:#172033; --muted:#637083; --surface:#fff; --line:#dce2ea; --official:#2563eb; --detected:#f59e0b; --missing:#dc2626; --extra:#8b5cf6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111827; --text:#e5e7eb; --muted:#a9b2c1; --surface:#182235; --line:#344156; --official:#60a5fa; --detected:#fbbf24; --missing:#f87171; --extra:#a78bfa; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:28px; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1180px; margin:auto; }}
h1 {{ margin:0 0 6px; font-size:26px; }}
h2 {{ margin:24px 0 8px; font-size:17px; }}
.intro,.metric {{ color:var(--muted); }}
.call-nav {{ display:flex; flex-wrap:wrap; gap:8px; margin:20px 0; }}
.call-button {{ border:1px solid var(--line); background:var(--surface); color:var(--text); border-radius:7px; padding:8px 10px; cursor:pointer; }}
.call-button.active {{ background:var(--text); color:var(--surface); border-color:var(--text); }}
.call-panel {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:20px; }}
.summary {{ display:flex; flex-wrap:wrap; gap:10px 24px; margin-bottom:12px; }}
audio {{ width:min(520px,100%); margin:2px 0 14px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 18px; color:var(--muted); }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.legend i {{ width:18px; height:8px; display:inline-block; border-radius:2px; }}
.legend .official,.segment.official {{ background:var(--official); }}
.legend .detected,.segment.detected {{ background:var(--detected); }}
.legend .missing,.segment.missing {{ background:var(--missing); }}
.legend .extra,.segment.extra {{ background:var(--extra); }}
.axis {{ position:relative; height:25px; margin-left:112px; border-bottom:1px solid var(--line); }}
.tick {{ position:absolute; bottom:-5px; width:1px; height:6px; background:var(--line); }}
.tick span {{ position:absolute; top:8px; transform:translateX(-50%); color:var(--muted); font-size:12px; white-space:nowrap; }}
.tick:first-child span {{ transform:none; }}
.tick:last-child span {{ transform:translateX(-100%); }}
.lane {{ display:grid; grid-template-columns:100px 1fr; gap:12px; align-items:center; margin:8px 0; }}
.lane-label {{ text-align:right; color:var(--muted); font-size:13px; }}
.track {{ position:relative; height:18px; background:color-mix(in srgb,var(--line) 45%,transparent); overflow:hidden; border-radius:3px; }}
.segment {{ position:absolute; inset-block:0; min-width:1px; }}
.metric {{ margin:7px 0 0 112px; font-size:13px; }}
@media (max-width:640px) {{ body {{ padding:14px; }} .call-panel {{ padding:14px; }} .axis {{ margin-left:82px; }} .lane {{ grid-template-columns:70px 1fr; gap:12px; }} .metric {{ margin-left:82px; }} }}
</style>
</head>
<body>
<main>
<h1>Errores de clasificación — Energy VAD</h1>
<p class="intro">Azul y amarillo deben verse similares. Rojo indica voz oficial que perdimos; morado indica voz adicional detectada.</p>
<nav class="call-nav" aria-label="Llamadas con error">{buttons}</nav>
{''.join(panels)}
</main>
<script>
const summaries={summary_json};
const buttons=[...document.querySelectorAll('.call-button')];
function showCall(callId) {{
  document.querySelectorAll('.call-panel').forEach(panel => panel.hidden = panel.id !== `panel-${{callId}}`);
  buttons.forEach(button => button.classList.toggle('active', button.dataset.call === callId));
}}
buttons.forEach(button => button.addEventListener('click', () => showCall(button.dataset.call)));
if (summaries.length) showCall(summaries[0].call_id);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--turns-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream) if row["label"] != row["prediction"]
        ]
    if not rows:
        raise SystemExit("No misclassified calls were found in the predictions CSV.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = VadConfig()
    panels, summaries = [], []
    for row in rows:
        panel, summary = build_panel(
            row, args.audio_dir, args.turns_dir, args.output.parent, config
        )
        panels.append(panel)
        summaries.append(summary)
    args.output.write_text(build_html(panels, summaries), encoding="utf-8")
    print(f"wrote {len(rows)} error timelines to {args.output}")


if __name__ == "__main__":
    main()
