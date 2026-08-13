"""A single self-contained HTML report — no external requests, no build
step, opens by double-clicking the file.

Visual language is a geological core log: each non-baseline language is a
drill core, baseline depth at the top, stages deposited as layers on the
way down to the language's actual score. A stage stratum can never be
measured never gets a layer sized from a number the method can't support —
see `_segments_for` and the "void" kind below, which is exactly the
`not measurable` rung `attribution.py` already refuses to fabricate,
rendered as a literal gap in the core rather than smoothed over.

Two tabs: Overview (a plain-English headline, score cards, the waterfalls,
a run-history strip, and a trust panel that sits beside the numbers, not
below them) and Detail (per-stage metric tables, failure taxonomy,
calibration table, latency, run metadata). All of it is generated from a
`Report` — the pydantic model, whether freshly produced by `Harness.run()`
or loaded back from a saved `report.json`. Nothing here reads
`Report.cascade_objects`: that field is excluded from JSON serialisation,
so code that depended on it would work for a fresh run and silently break
for `stratum html <report.json>`.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path

from .report import Report

# ---------------------------------------------------------------------------
# Palette — validated with the dataviz skill's validate_palette.js against a
# custom dark mineral ground, not eyeballed. The 4 stage hues pass the
# adjacent-pair CVD/lightness/chroma/contrast gates for a stacked-bar
# context (worst adjacent ΔE 8.6 deutan, 21.3 normal-vision); status and
# accent are checked individually for ≥3:1 contrast against the same
# ground, since they never compete with the stage hues as identity marks
# in the same chart.
# ---------------------------------------------------------------------------

PALETTE = {
    "page": "#0f0e0c",
    "surface": "#1a1815",
    "surface_raised": "#232019",
    "surface_sunken": "#141210",
    "border": "rgba(243,240,234,0.10)",
    "border_strong": "rgba(243,240,234,0.18)",
    "grid": "#2c291f",
    "ink": "#f3f0ea",
    "ink_secondary": "#b6afa1",
    "ink_muted": "#847c6d",
    "accent": "#e8b73a",
    "accent_ink": "#3a2c05",
    "good": "#4fa88a",
    "warning": "#d68a3f",
    "critical": "#c1503f",
}

#: One hue per cascade stage key, fixed order, never cycled.
STAGE_COLOR = {
    "s0_s1_input_query": "#c17f4a",  # clay
    "s2_retrieval": "#3d76b0",       # slate
    "s3_generation": "#7a9440",      # moss
    "s4_rendering": "#b85a78",       # mauve
}

LANGUAGE_NAMES = {
    "en": "English", "hi-Deva": "Hindi (Devanagari)", "hi-Latn": "Roman-Hindi",
    "ta-Taml": "Tamil", "bn-Beng": "Bengali", "gu-Gujr": "Gujarati",
    "kn-Knda": "Kannada", "ml-Mlym": "Malayalam", "mr-Deva": "Marathi",
    "pa-Guru": "Punjabi", "te-Telu": "Telugu", "ur-Arab": "Urdu",
    "es": "Spanish", "fr": "French", "de": "German", "ja": "Japanese",
    "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese", "ru": "Russian",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def stage_colors_for(stage_key: str) -> list[str]:
    """One hex per component stage — usually one, two for a combined band
    ("s2_retrieval+s3_generation")."""
    return [STAGE_COLOR.get(part, PALETTE["ink_muted"]) for part in stage_key.split("+")]


def _esc(value) -> str:
    return _html.escape(str(value), quote=True)


def _fmt(value: float | None, decimals: int = 1) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Overview: headline
# ---------------------------------------------------------------------------

def build_headline(report: Report) -> str:
    """One plain-English sentence naming the worst finding this run — the
    thing someone who won't read the code needs to walk away knowing."""
    scored = [c for c in report.cascades if c.get("total_loss") is not None]
    if not scored:
        return f"{report.system_label}: no cross-language comparison available for this run."

    worst = max(scored, key=lambda c: c["total_loss"])
    if worst["total_loss"] <= 0:
        return (
            f"No language scored below {language_name(report.baseline_language)} "
            f"this run — the widest gap was {language_name(worst['language'])} at "
            f"{worst['total_loss']:+.0f} points."
        )

    lang = language_name(worst["language"])
    baseline = language_name(worst["baseline_language"])
    total = worst["total_loss"]

    dom_key = worst.get("dominant_stage")
    dom = next((s for s in worst["by_stage"] if s["stage"] == dom_key), None) if dom_key else None

    if dom is None:
        return (
            f"{lang} scored {total:.0f} points below {baseline} — no single stage "
            f"explains it at this sample size."
        )

    has_unmeasurable = any(s["points_lost"] is None for s in worst["by_stage"])
    if has_unmeasurable:
        denom = sum(
            s["points_lost"] for s in worst["by_stage"]
            if s["points_lost"] is not None and s["points_lost"] > 0
        )
        qualifier = "of measured loss"
    else:
        denom = total
        qualifier = "of it"
    share = (dom["points_lost"] / denom * 100) if denom else 0.0

    return (
        f"{lang} lost {total:.0f} points against {baseline} — {share:.0f}% "
        f"{qualifier} at {dom['label'].lower()}."
    )


# ---------------------------------------------------------------------------
# Overview: language cards
# ---------------------------------------------------------------------------

def language_cards(report: Report, target_delta: float = -5.0) -> list[dict]:
    """Mirrors Report.render_terminal's three-way status logic exactly —
    same data, same conclusion, two presentations. A card's status must
    never disagree with what the terminal output already told this user."""
    cards = []
    for lr in report.languages:
        is_baseline = lr.language == report.baseline_language
        delta = lr.delta_vs_baseline
        delta_ci = lr.delta.get("ci95") if lr.delta else None

        if is_baseline:
            status = "baseline"
        elif delta is None or not delta_ci or delta_ci[0] is None:
            status = "unknown"
        elif delta_ci[1] < 0 and delta < target_delta:
            status = "critical"
        elif delta_ci[0] <= 0 <= delta_ci[1]:
            status = "noise"
        else:
            status = "good"

        cards.append({
            "language": lr.language,
            "name": language_name(lr.language),
            "score": lr.quality.get("value"),
            "ci": lr.quality.get("ci95"),
            "n": lr.n_items,
            "delta": delta,
            "delta_ci": delta_ci,
            "is_baseline": is_baseline,
            "status": status,
            "verified": lr.verified,
        })
    return cards


# ---------------------------------------------------------------------------
# Run history: sibling reports/*/report.json
# ---------------------------------------------------------------------------

def scan_report_history(reports_dir: Path, *, exclude: Path | None = None) -> list[dict]:
    """Every sibling report this run can be compared against, oldest first.

    Malformed or unreadable entries are skipped, not fatal — a report a
    previous stratum version wrote (or a directory that isn't a report at
    all) must not crash the history strip for every run after it.
    """
    reports_dir = Path(reports_dir)
    if not reports_dir.is_dir():
        return []

    entries = []
    for candidate in sorted(reports_dir.glob("*/report.json")):
        if exclude is not None and candidate.parent == Path(exclude):
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            entries.append({
                "path": str(candidate),
                "generated_at": data.get("generated_at", ""),
                "system_label": data.get("system_label", ""),
                "languages": {
                    lr["language"]: lr.get("quality", {}).get("value")
                    for lr in data.get("languages", [])
                },
            })
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            continue

    entries.sort(key=lambda e: e["generated_at"])
    return entries


# ---------------------------------------------------------------------------
# Waterfall: the core-log centrepiece
# ---------------------------------------------------------------------------

#: Depth axis scale — fixed across every report, deliberately not rescaled
#: to fit one run's data. A rescaled axis would make two runs' waterfalls
#: visually incomparable at a glance, defeating the run-history strip's
#: whole point.
PX_PER_POINT = 3.6
TUBE_W = 84
TUBE_GAP = 72
TOP_PAD = 30
BOTTOM_PAD = 46
MIN_SEGMENT_PX = 5


def _segments_for(cascade: dict) -> list[dict]:
    """Turn `by_stage` into drawable depth segments.

    A contiguous run of unmeasured (`points_lost: None`) stages becomes one
    "void" segment, sized by what's left after every measured stage is
    accounted for (`total_loss` minus the measured sum) — an honestly
    *derived* number, not a fabricated per-stage one; attribution.py can
    tell you S2+S3 cost this many points combined without ever claiming to
    know the split, and the void's rendering (no fill, dashed, hatched)
    keeps that non-claim visible rather than smoothing it into a number
    that looks like any other measured rung.
    """
    stages = cascade["by_stage"]
    total_loss = cascade.get("total_loss")
    measured_sum = sum(s["points_lost"] for s in stages if s["points_lost"] is not None)
    remainder = (total_loss - measured_sum) if total_loss is not None else None

    segments: list[dict] = []
    cursor = 0.0
    i, n = 0, len(stages)
    while i < n:
        s = stages[i]
        if s["points_lost"] is not None:
            points = s["points_lost"]
            segments.append({
                "kind": "measured",
                "stages": [s],
                "depth_start": cursor,
                "depth_end": cursor + points,
                "points": points,
                "noise": s["indistinguishable_from_zero"],
                "negative": points < 0,
                "combined": not s["isolated"],
                "note": s.get("note", ""),
            })
            cursor += points
            i += 1
        else:
            run = []
            while i < n and stages[i]["points_lost"] is None:
                run.append(stages[i])
                i += 1
            size = remainder if (remainder is not None and remainder > 0) else 8.0
            segments.append({
                "kind": "void", "stages": run,
                "depth_start": cursor, "depth_end": cursor + size,
                "points": remainder,
            })
            cursor += size
    return segments


def _stage_pattern_id(hex_color: str) -> str:
    return f"hatch-{hex_color.lstrip('#')}"


def _hatch_defs(hexes: set[str]) -> str:
    """One diagonal-line pattern per stage hue, for the `noise` state — the
    same tone-on-tone 45° texture channel the dataviz skill specifies,
    reused here as "measured, but not distinguishable from zero" rather
    than a CVD fallback."""
    defs = ['<pattern id="hatch-void" width="10" height="10" patternTransform="rotate(45)" '
            'patternUnits="userSpaceOnUse"><line x1="0" y1="0" x2="0" y2="10" '
            f'stroke="{PALETTE["ink_muted"]}" stroke-width="1" opacity="0.35"/></pattern>']
    for hexcolor in sorted(hexes):
        defs.append(
            f'<pattern id="{_stage_pattern_id(hexcolor)}" width="9" height="9" '
            f'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
            f'<rect width="9" height="9" fill="{hexcolor}" fill-opacity="0.28"/>'
            f'<line x1="0" y1="0" x2="0" y2="9" stroke="{hexcolor}" stroke-width="2.5" opacity="0.55"/>'
            f'</pattern>'
        )
    return "".join(defs)


def render_waterfall_svg(cascades: list[dict]) -> str:
    """All languages' core samples, worst (longest) first, sharing one
    points-lost depth axis so tube length is directly comparable."""
    ordered = sorted(
        (c for c in cascades if c.get("total_loss") is not None),
        key=lambda c: c["total_loss"], reverse=True,
    )
    if not ordered:
        return '<p class="empty-note">No cascade data available for this run.</p>'

    per_lang_segments = {c["language"]: _segments_for(c) for c in ordered}
    max_depth = 10.0
    for segs in per_lang_segments.values():
        for seg in segs:
            max_depth = max(max_depth, abs(seg["depth_start"]), abs(seg["depth_end"]))

    used_hexes = {
        hexc
        for segs in per_lang_segments.values()
        for seg in segs if seg["kind"] == "measured"
        for hexc in stage_colors_for(seg["stages"][0]["stage"])
    }

    height = TOP_PAD + max_depth * PX_PER_POINT + BOTTOM_PAD
    width = len(ordered) * (TUBE_W + TUBE_GAP) + TUBE_GAP

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'preserveAspectRatio="xMidYMin meet" role="img" '
        f'aria-label="Degradation cascade, one core sample per language">',
        f"<defs>{_hatch_defs(used_hexes)}</defs>",
    ]

    # -- shared depth gridlines (points lost from the surface) -------------
    step = 10 if max_depth <= 60 else 20
    d = 0
    while d <= max_depth:
        y = TOP_PAD + d * PX_PER_POINT
        parts.append(
            f'<line x1="0" y1="{y:.1f}" x2="{width:.0f}" y2="{y:.1f}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1"/>'
            f'<text x="8" y="{y - 4:.1f}" class="wf-depth-label">−{d:.0f}</text>'
        )
        d += step

    x = TUBE_GAP
    for cascade in ordered:
        parts.append(_render_tube(cascade, per_lang_segments[cascade["language"]], x))
        x += TUBE_W + TUBE_GAP

    parts.append("</svg>")
    return "".join(parts)


def _render_tube(cascade: dict, segments: list[dict], x: float) -> str:
    lang = cascade["language"]
    dominant = cascade.get("dominant_stage")
    g = [f'<g class="wf-tube" data-language="{_esc(lang)}">']

    # surface cap — baseline reference
    g.append(
        f'<line x1="{x:.1f}" y1="{TOP_PAD:.1f}" x2="{x + TUBE_W:.1f}" y2="{TOP_PAD:.1f}" '
        f'stroke="{PALETTE["ink_secondary"]}" stroke-width="2"/>'
    )

    for seg in segments:
        y0 = TOP_PAD + min(seg["depth_start"], seg["depth_end"]) * PX_PER_POINT
        y1 = TOP_PAD + max(seg["depth_start"], seg["depth_end"]) * PX_PER_POINT
        seg_h = max(y1 - y0, MIN_SEGMENT_PX)
        is_dominant = seg["kind"] == "measured" and seg["stages"][0]["stage"] == dominant

        if seg["kind"] == "void":
            stage_names = " + ".join(s["label"] for s in seg["stages"])
            g.append(
                f'<rect x="{x + 1:.1f}" y="{y0:.1f}" width="{TUBE_W - 2:.1f}" height="{seg_h:.1f}" '
                f'fill="url(#hatch-void)" stroke="{PALETTE["ink_muted"]}" stroke-width="1.5" '
                f'stroke-dasharray="4 3" rx="2">'
                f'<title>{_esc(stage_names)} — not measured (no calibrated judge)</title></rect>'
            )
            if seg_h > 22:
                g.append(
                    f'<text x="{x + TUBE_W / 2:.1f}" y="{(y0 + y1) / 2:.1f}" '
                    f'class="wf-void-label">not measured</text>'
                )
            continue

        stage = seg["stages"][0]
        hexes = stage_colors_for(stage["stage"])
        fill = hexes[0]
        pattern = f'url(#{_stage_pattern_id(fill)})' if seg["noise"] else fill
        classes = "wf-seg"
        if is_dominant:
            classes += " wf-seg-dominant"

        # Negated for the same reason as the tube total: points_lost is
        # positive-means-worse, and attribution.py's own terminal renderer
        # negates every per-rung value for display, not just the header.
        title = (
            f"{stage['label']}: {-seg['points']:+.1f} pts"
            if seg["points"] is not None else stage["label"]
        )
        if seg["noise"]:
            title += " (noise — indistinguishable from zero)"
        if seg["combined"]:
            title += " (combined band, not isolated)"
        if seg["negative"]:
            title += " (negative — repairing this hurt)"

        g.append(
            f'<rect class="{classes}" x="{x:.1f}" y="{y0:.1f}" width="{TUBE_W:.1f}" '
            f'height="{seg_h:.1f}" fill="{pattern}"'
            + (f' stroke="{PALETTE["accent"]}" stroke-width="2.5"' if is_dominant else "")
            + (f' stroke-dasharray="3 2" stroke="{PALETTE["good"]}" stroke-width="2"'
               if seg["negative"] and not is_dominant else "")
            + f'><title>{_esc(title)}</title></rect>'
        )
        if seg["combined"]:
            g.append(
                f'<rect x="{x:.1f}" y="{y0:.1f}" width="{TUBE_W:.1f}" height="{seg_h:.1f}" '
                f'fill="none" stroke="{PALETTE["page"]}" stroke-width="1.5" '
                f'stroke-dasharray="1 3" opacity="0.6"/>'
            )

    total_depth = segments[-1]["depth_end"] if segments else 0
    bottom_y = TOP_PAD + total_depth * PX_PER_POINT
    g.append(
        f'<text x="{x + TUBE_W / 2:.1f}" y="{bottom_y + 18:.1f}" class="wf-lang-label">'
        f'{_esc(language_name(lang))}</text>'
        f'<text x="{x + TUBE_W / 2:.1f}" y="{bottom_y + 33:.1f}" class="wf-total-label">'
        # total_loss is positive-means-worse; negate before the leading '+'
        # or a loss reads as a gain — see attribution.py's own sign-bug note.
        f'{-cascade["total_loss"]:+.0f} pts</text>'
    )
    g.append("</g>")
    return "".join(g)


# ---------------------------------------------------------------------------
# Overview: HTML fragments
# ---------------------------------------------------------------------------

_STATUS_ICON = {"good": "✓", "noise": "≈", "critical": "✕", "baseline": "◆", "unknown": "?"}
_STATUS_TEXT = {
    "good": "clears target", "noise": "indistinguishable from baseline",
    "critical": "over target", "baseline": "baseline", "unknown": "no comparison",
}


def render_cards_html(cards: list[dict]) -> str:
    out = ['<div class="cards">']
    for c in cards:
        ci_html = ""
        if c["ci"] and c["ci"][0] is not None:
            width = (c["ci"][1] - c["ci"][0]) / 2
            ci_html = f'<span class="card-ci">±{width:.1f}</span>'

        delta_html = "baseline"
        if not c["is_baseline"] and c["delta"] is not None:
            delta_html = f'{c["delta"]:+.1f} pts vs baseline'
            if c["delta_ci"] and c["delta_ci"][0] is not None:
                delta_html += (
                    f'<span class="card-ci-range">'
                    f'[{c["delta_ci"][0]:+.1f}, {c["delta_ci"][1]:+.1f}]</span>'
                )

        experimental = "" if c["verified"] else '<span class="chip chip-muted">experimental</span>'

        out.append(f'''
        <div class="card card-{c["status"]}">
          <div class="card-top">
            <span class="card-icon" aria-hidden="true">{_STATUS_ICON[c["status"]]}</span>
            <span class="card-lang">{_esc(c["name"])}</span>
            {experimental}
          </div>
          <div class="card-score">{_fmt(c["score"], 1)}{ci_html}</div>
          <div class="card-delta">{delta_html}</div>
          <div class="card-n">n={c["n"]}</div>
          <div class="card-status-text">{_STATUS_TEXT[c["status"]]}</div>
        </div>''')
    out.append("</div>")
    return "".join(out)


def render_history_strip_html(history: list[dict], current: dict | None = None) -> str:
    """A sparkline row of prior runs, so improvement across runs is visible
    at a glance — not just this one run's numbers."""
    runs = list(history)
    if current is not None:
        runs = runs + [current]
    if len(runs) < 2:
        return (
            '<div class="history-strip history-empty">'
            "No prior runs found in this reports directory yet — history will "
            "appear here once there are more to compare."
            "</div>"
        )

    langs = sorted({lang for r in runs for lang in r["languages"]})
    max_score = 100.0
    rows = []
    for lang in langs:
        points = [r["languages"].get(lang) for r in runs]
        if all(p is None for p in points):
            continue
        pts_html = []
        n = len(runs)
        for i, (r, v) in enumerate(zip(runs, points)):
            x = (i / max(n - 1, 1)) * 100
            label = r.get("system_label") or r.get("generated_at", "")
            if v is None:
                continue
            y = 100 - (v / max_score) * 100
            pts_html.append(f'<circle cx="{x:.1f}%" cy="{y:.1f}%" r="3.5" '
                             f'class="hist-dot"><title>{_esc(label)}: {v:.1f}</title></circle>')
        path_pts = [
            (i / max(n - 1, 1)) * 100
            for i in range(n)
        ]
        line = " ".join(
            f'{px:.1f},{100 - (v / max_score) * 100:.1f}'
            for px, v in zip(path_pts, points) if v is not None
        )
        rows.append(f'''
        <div class="history-row">
          <span class="history-lang">{_esc(language_name(lang))}</span>
          <svg class="history-sparkline" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polyline points="{line}" class="hist-line" vector-effect="non-scaling-stroke"/>
          </svg>
          <svg class="history-dots" viewBox="0 0 100 100" preserveAspectRatio="none">
            {"".join(pts_html)}
          </svg>
        </div>''')

    labels = [r.get("system_label") or "run" for r in runs]
    return f'''
    <div class="history-strip">
      <div class="history-header">
        <span>RUN HISTORY</span>
        <span class="history-range">{_esc(labels[0])} → {_esc(labels[-1])} · {len(runs)} runs</span>
      </div>
      {"".join(rows)}
    </div>'''


def _trust_row(label: str, value: str, *, warn: bool = False, wide: bool = False) -> str:
    classes = "trust-item"
    if warn:
        classes += " trust-warn"
    if wide:
        classes += " trust-item-wide"
    return (
        f'<div class="{classes}"><span class="trust-label">{_esc(label)}</span>'
        f'<span class="trust-value">{_esc(value)}</span></div>'
    )


def trust_panel_html(report: Report) -> str:
    """Caveats beside the numbers, not below them — every reason to
    distrust a figure on this page, in one place on the Overview tab.

    Short numeric facts (n, interval width) sit in the narrow grid columns;
    anything sentence-length gets `wide` — a full-width row, left-aligned —
    so a real caveat doesn't get squeezed into a ragged, hard-to-read column
    the way a short "n=23" value never does.
    """
    items: list[str] = []

    for lr in report.languages:
        ci = lr.quality.get("ci95")
        width = f"±{(ci[1] - ci[0]) / 2:.1f}" if ci and ci[0] is not None else "—"
        items.append(_trust_row(language_name(lr.language), f"n={lr.n_items} · interval {width} pts"))

    if report.calibrations:
        untrustworthy = [c for c in report.calibrations if not c["trustworthy"]]
        if untrustworthy:
            names = ", ".join(f"{c['language']}/{c['metric']} (κ={c['kappa']})" for c in untrustworthy)
            items.append(_trust_row("Judge calibration", f"below threshold: {names}", warn=True, wide=True))
        trustworthy = [c for c in report.calibrations if c["trustworthy"]]
        if trustworthy:
            names = ", ".join(f"{c['language']}/{c['metric']}" for c in trustworthy)
            items.append(_trust_row("Judge calibration", f"trustworthy: {names}", wide=True))
    else:
        note = next(
            (s["note"] for c in report.cascades for s in c["by_stage"]
             if s["points_lost"] is None and s.get("note")),
            None,
        )
        if note:
            items.append(_trust_row("Judge calibration", note, warn=True, wide=True))

    for w in report.warnings:
        items.append(_trust_row("Caveat", w, warn=True, wide=True))

    return f'''
    <div class="trust-panel">
      <div class="trust-header">TRUST</div>
      <div class="trust-grid">{"".join(items)}</div>
    </div>'''


# ---------------------------------------------------------------------------
# Detail tab
# ---------------------------------------------------------------------------

#: Which stage each deterministic metric belongs to, for the Detail tab's
#: per-stage tables — a diagnostic grouping only; it has no bearing on
#: OUTCOME_WEIGHTS or the cascade math in harness.py.
METRIC_STAGE = {
    "language_detection": ("s0_s1_input_query", "S0+S1 · Input & query"),
    "recall_at_k": ("s2_retrieval", "S2 · Retrieval"),
    "mrr": ("s2_retrieval", "S2 · Retrieval"),
    "ndcg_at_k": ("s2_retrieval", "S2 · Retrieval"),
    "over_refusal": ("s3_generation", "S3 · Generation"),
    "answered_correctly": ("s3_generation", "S3 · Generation"),
    "faithfulness": ("s3_generation", "S3 · Generation"),
    "answer_correctness": ("s3_generation", "S3 · Generation"),
    "placeholder_integrity": ("s4_rendering", "S4 · Rendering"),
    "numeral_integrity": ("s4_rendering", "S4 · Rendering"),
    "entity_preservation": ("s4_rendering", "S4 · Rendering"),
    "glossary_adherence": ("s4_rendering", "S4 · Rendering"),
}
METRIC_LABEL = {
    "language_detection": "Language detection", "recall_at_k": "Recall@k",
    "mrr": "MRR", "ndcg_at_k": "nDCG@k", "over_refusal": "Over-refusal",
    "answered_correctly": "Answered correctly", "faithfulness": "Faithfulness",
    "answer_correctness": "Answer correctness", "placeholder_integrity": "Placeholder integrity",
    "numeral_integrity": "Numeral integrity", "entity_preservation": "Entity preservation",
    "glossary_adherence": "Glossary adherence",
}


def detail_metrics_html(report: Report) -> str:
    """One table per language, metrics grouped by the stage they diagnose —
    value, CI, and n for every metric that was actually observed."""
    sections = []
    for lr in report.languages:
        stage_groups: dict[str, list] = {}
        for key, m in lr.metrics.items():
            if not isinstance(m, dict) or m.get("n", 0) == 0:
                continue
            stage_key, stage_label = METRIC_STAGE.get(key, ("other", "Other"))
            stage_groups.setdefault(stage_label, []).append((key, m))

        if not stage_groups:
            continue

        rows = []
        for stage_label in sorted(stage_groups):
            for key, m in stage_groups[stage_label]:
                ci = m.get("ci95")
                ci_text = f'[{ci[0]:+.1f}, {ci[1]:+.1f}]' if ci and ci[0] is not None else "—"
                precise = "" if m.get("precise") else '<span class="chip chip-muted">wide CI</span>'
                rows.append(
                    f'<tr><td class="td-stage">{_esc(stage_label)}</td>'
                    f'<td>{_esc(METRIC_LABEL.get(key, key))}</td>'
                    f'<td class="td-num">{_fmt(m.get("value"))}</td>'
                    f'<td class="td-num">{ci_text}</td>'
                    f'<td class="td-num">{m.get("n", 0)}</td>'
                    f'<td>{precise}</td></tr>'
                )

        sections.append(f'''
        <div class="detail-block">
          <h3>{_esc(language_name(lr.language))}</h3>
          <table class="detail-table">
            <thead><tr><th>Stage</th><th>Metric</th><th>Value</th><th>95% CI</th><th>n</th><th></th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>''')
    return "".join(sections) or '<p class="empty-note">No per-item metrics recorded.</p>'


def detail_taxonomy_html(report: Report) -> str:
    """Failure counts, each expandable (native <details>, no JS needed) into
    the individual examples that made up that count."""
    if not report.taxonomy:
        return '<p class="empty-note">No failures recorded.</p>'

    by_cls: dict[str, list] = {}
    for f in report.failures:
        by_cls.setdefault(f.cls, []).append(f)

    blocks = []
    for cls, count in report.taxonomy.items():
        examples = by_cls.get(cls, [])[:8]
        rows = "".join(
            f'<tr><td>{_esc(f.item_id)}</td><td>{_esc(f.language)}</td>'
            f'<td>{_esc(f.stage)}</td><td class="td-query">{_esc(f.query[:80])}</td>'
            f'<td class="td-query">{_esc(f.detail[:100])}</td></tr>'
            for f in examples
        )
        blocks.append(f'''
        <details class="taxonomy-group">
          <summary><span class="taxonomy-cls">{_esc(cls)}</span>
            <span class="taxonomy-count">{count}</span></summary>
          <table class="detail-table taxonomy-table">
            <thead><tr><th>Item</th><th>Language</th><th>Stage</th><th>Query</th><th>Detail</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </details>''')
    return "".join(blocks)


def detail_calibration_html(report: Report) -> str:
    if not report.calibrations:
        return (
            '<p class="empty-note">No calibration loaded this run — S2/S3 judged '
            "metrics stay not measurable. Run <code>stratum calibrate</code> to "
            "produce one.</p>"
        )
    rows = []
    for c in sorted(report.calibrations, key=lambda c: (c["language"], c["metric"])):
        status = "trustworthy" if c["trustworthy"] else "below threshold"
        cls = "status-good" if c["trustworthy"] else "status-warn"
        rows.append(
            f'<tr><td>{_esc(language_name(c["language"]))}</td><td>{_esc(c["metric"])}</td>'
            f'<td class="td-num">{c["n_labelled"]}</td><td class="td-num">{c["kappa"]:.3f}</td>'
            f'<td class="{cls}">{status}</td><td>{_esc(c["judge_id"])}</td></tr>'
        )
    return f'''
    <table class="detail-table">
      <thead><tr><th>Language</th><th>Metric</th><th>n labelled</th><th>κ</th>
        <th>Status</th><th>Judge</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>'''


def detail_latency_html(report: Report) -> str:
    rows = "".join(
        f'<tr><td>{_esc(language_name(lr.language))}</td>'
        f'<td class="td-num">{_fmt(lr.latency_p50, 0)} ms</td>'
        f'<td class="td-num">{_fmt(lr.latency_p90, 0)} ms</td></tr>'
        for lr in report.languages
    )
    return f'''
    <table class="detail-table">
      <thead><tr><th>Language</th><th>Latency p50</th><th>Latency p90</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p class="empty-note">End-to-end only — no per-stage timing breakdown or cost
      is instrumented by this endpoint for this run.</p>'''


def detail_metadata_html(report: Report) -> str:
    rows = [
        ("System", report.system_label),
        ("Generated", report.generated_at),
        ("Baseline language", language_name(report.baseline_language)),
        ("Items", str(report.n_items)),
        ("Passes run", ", ".join(report.passes_run)),
        ("Dataset hash", report.dataset_hash or "—"),
        ("Config hash", report.config_hash or "—"),
    ]
    for name, version in report.model_versions.items():
        rows.append((f"Model — {name}", version))
    rows_html = "".join(
        f'<tr><td class="td-meta-key">{_esc(k)}</td><td><code>{_esc(v)}</code></td></tr>'
        for k, v in rows
    )
    return f'<table class="detail-table"><tbody>{rows_html}</tbody></table>'


# ---------------------------------------------------------------------------
# Page shell — CSS + JS, both inline. No external requests of any kind.
# ---------------------------------------------------------------------------

def _css() -> str:
    p = PALETTE
    return f'''
    :root {{
      --page: {p["page"]}; --surface: {p["surface"]}; --surface-raised: {p["surface_raised"]};
      --surface-sunken: {p["surface_sunken"]}; --border: {p["border"]}; --border-strong: {p["border_strong"]};
      --grid: {p["grid"]}; --ink: {p["ink"]}; --ink-secondary: {p["ink_secondary"]};
      --ink-muted: {p["ink_muted"]}; --accent: {p["accent"]}; --accent-ink: {p["accent_ink"]};
      --good: {p["good"]}; --warning: {p["warning"]}; --critical: {p["critical"]};
      --s0: {STAGE_COLOR["s0_s1_input_query"]}; --s2: {STAGE_COLOR["s2_retrieval"]};
      --s3: {STAGE_COLOR["s3_generation"]}; --s4: {STAGE_COLOR["s4_rendering"]};
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ background: var(--page); color: var(--ink); margin: 0; padding: 0; }}
    body {{
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      font-size: 15px; line-height: 1.5; -webkit-font-smoothing: antialiased;
    }}
    code {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.9em; }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 0 28px 80px; }}

    /* -- masthead --------------------------------------------------- */
    .masthead {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 28px 0 18px; border-bottom: 1px solid var(--border);
      flex-wrap: wrap; gap: 14px;
    }}
    .brand {{ display: flex; align-items: baseline; gap: 12px; }}
    .brand-mark {{ font-weight: 700; letter-spacing: 0.14em; font-size: 13px; color: var(--accent); }}
    .brand-meta {{ color: var(--ink-muted); font-size: 13px; }}
    .tabs {{ display: flex; gap: 4px; background: var(--surface); border-radius: 8px; padding: 3px; }}
    .tab-btn {{
      appearance: none; border: none; background: transparent; color: var(--ink-secondary);
      font: inherit; font-size: 13px; font-weight: 600; padding: 8px 18px; border-radius: 6px;
      cursor: pointer; letter-spacing: 0.02em;
    }}
    .tab-btn.active {{ background: var(--surface-raised); color: var(--accent); }}
    .tab-btn:hover:not(.active) {{ color: var(--ink); }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

    /* -- hero: headline + cards + waterfall -------------------------- */
    .hero {{ padding-top: 22px; }}
    .headline {{
      font-size: 28px; font-weight: 650; line-height: 1.35; letter-spacing: -0.01em;
      margin: 0 0 20px; max-width: 46ch;
    }}
    .headline .accent {{ color: var(--accent); }}

    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 24px; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 15px 18px; border-left: 3px solid var(--ink-muted);
    }}
    .card-baseline {{ border-left-color: var(--ink-secondary); }}
    .card-good {{ border-left-color: var(--good); }}
    .card-noise {{ border-left-color: var(--ink-muted); }}
    .card-critical {{ border-left-color: var(--critical); }}
    .card-top {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
    .card-icon {{ font-size: 13px; color: var(--ink-muted); }}
    .card-good .card-icon {{ color: var(--good); }}
    .card-critical .card-icon {{ color: var(--critical); }}
    .card-lang {{ font-weight: 600; font-size: 14px; }}
    .card-score {{ font-size: 34px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; }}
    .card-ci {{ font-size: 15px; color: var(--ink-muted); font-weight: 500; margin-left: 6px; }}
    .card-delta {{ margin-top: 8px; font-size: 13px; color: var(--ink-secondary); }}
    .card-ci-range {{ color: var(--ink-muted); margin-left: 6px; }}
    .card-n {{ margin-top: 10px; font-size: 12px; color: var(--ink-muted); font-variant-numeric: tabular-nums; }}
    .card-status-text {{ margin-top: 2px; font-size: 11px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .chip {{
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; padding: 2px 7px;
      border-radius: 20px; background: var(--surface-raised); color: var(--ink-muted);
      border: 1px solid var(--border);
    }}
    .chip-muted {{ color: var(--warning); border-color: color-mix(in srgb, var(--warning) 40%, transparent); }}

    .waterfall-block {{
      background: var(--surface-sunken); border: 1px solid var(--border); border-radius: 14px;
      padding: 18px 20px 12px; margin-bottom: 44px;
    }}
    .waterfall-block svg {{ display: block; margin: 0 auto; max-width: 100%; height: auto; }}
    .wf-depth-label {{ fill: var(--ink-muted); font-size: 10px; font-variant-numeric: tabular-nums; }}
    .wf-void-label {{ fill: var(--ink-muted); font-size: 10.5px; text-anchor: middle; }}
    .wf-lang-label {{ fill: var(--ink); font-size: 13px; font-weight: 600; text-anchor: middle; }}
    .wf-total-label {{ fill: var(--ink-secondary); font-size: 11.5px; text-anchor: middle; font-variant-numeric: tabular-nums; }}
    .wf-seg {{ transition: opacity 0.15s; }}
    .wf-seg:hover {{ opacity: 0.85; }}
    .waterfall-legend {{ display: flex; flex-wrap: wrap; gap: 16px; padding: 4px 8px 18px; font-size: 12px; color: var(--ink-secondary); }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend-swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
    .legend-swatch-void {{ border: 1.5px dashed var(--ink-muted); background: transparent; }}

    /* -- history strip ------------------------------------------------ */
    .history-strip {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin-bottom: 28px; }}
    .history-empty {{ color: var(--ink-muted); font-size: 13px; padding: 22px; text-align: center; }}
    .history-header {{ display: flex; justify-content: space-between; font-size: 11px; letter-spacing: 0.06em;
      color: var(--ink-muted); margin-bottom: 14px; }}
    .history-range {{ color: var(--ink-muted); font-weight: 400; letter-spacing: 0; }}
    .history-row {{ display: grid; grid-template-columns: 130px 1fr; align-items: center; gap: 14px; height: 30px; position: relative; }}
    .history-lang {{ font-size: 12.5px; color: var(--ink-secondary); }}
    .history-sparkline, .history-dots {{ position: relative; grid-column: 2; width: 100%; height: 26px; overflow: visible; }}
    .history-dots {{ position: absolute; right: 0; width: 100%; }}
    .hist-line {{ fill: none; stroke: var(--accent); stroke-width: 1.6; opacity: 0.8; }}
    .hist-dot {{ fill: var(--accent); }}

    /* -- trust panel --------------------------------------------------- */
    .trust-panel {{ background: var(--surface); border: 1px solid var(--border-strong); border-radius: 12px; padding: 20px 22px; margin-bottom: 20px; }}
    .trust-header {{ font-size: 11px; letter-spacing: 0.1em; color: var(--accent); margin-bottom: 14px; font-weight: 700; }}
    .trust-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px 24px; }}
    .trust-item {{ display: flex; justify-content: space-between; gap: 14px; font-size: 12.5px; padding: 7px 0; border-bottom: 1px solid var(--border); }}
    .trust-label {{ color: var(--ink-secondary); flex-shrink: 0; }}
    .trust-value {{ color: var(--ink-muted); text-align: right; }}
    .trust-warn .trust-value {{ color: var(--warning); }}
    .trust-item-wide {{
      grid-column: 1 / -1; flex-direction: column; align-items: flex-start; gap: 4px;
    }}
    .trust-item-wide .trust-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ink-muted); }}
    .trust-item-wide .trust-value {{ text-align: left; max-width: 80ch; line-height: 1.5; }}

    /* -- detail tab ----------------------------------------------------- */
    .detail-section {{ margin: 44px 0; }}
    .detail-section h2 {{ font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); margin: 0 0 16px; }}
    .detail-block {{ margin-bottom: 26px; }}
    .detail-block h3 {{ font-size: 14px; margin: 0 0 10px; color: var(--ink-secondary); font-weight: 600; }}
    .detail-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .detail-table th {{
      text-align: left; font-weight: 600; color: var(--ink-muted); font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.04em; padding: 8px 10px; border-bottom: 1px solid var(--border-strong);
    }}
    .detail-table td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--ink-secondary); }}
    .detail-table tr:hover td {{ background: var(--surface); }}
    .td-num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .td-stage {{ color: var(--ink-muted); font-size: 12px; }}
    .td-query {{ color: var(--ink-muted); max-width: 320px; }}
    .td-meta-key {{ color: var(--ink-muted); width: 220px; }}
    .status-good {{ color: var(--good); }}
    .status-warn {{ color: var(--warning); }}

    .taxonomy-group {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 10px; overflow: hidden; }}
    .taxonomy-group summary {{
      cursor: pointer; padding: 13px 18px; display: flex; align-items: center; gap: 12px;
      list-style: none; font-size: 13.5px; user-select: none;
    }}
    .taxonomy-group summary::-webkit-details-marker {{ display: none; }}
    .taxonomy-group summary::before {{ content: "▸"; color: var(--ink-muted); font-size: 11px; transition: transform 0.15s; }}
    .taxonomy-group[open] summary::before {{ transform: rotate(90deg); }}
    .taxonomy-cls {{ font-weight: 600; }}
    .taxonomy-count {{ color: var(--warning); font-variant-numeric: tabular-nums; margin-left: auto; padding-right: 8px; }}
    .taxonomy-table {{ padding: 0 14px 14px; }}

    .empty-note {{ color: var(--ink-muted); font-size: 13px; font-style: italic; }}

    footer.page-footer {{ margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--ink-muted); font-size: 11.5px; }}

    @media (max-width: 640px) {{
      .headline {{ font-size: 22px; }}
      .history-row {{ grid-template-columns: 90px 1fr; }}
      .trust-grid {{ grid-template-columns: 1fr; }}
    }}
    '''


def _js() -> str:
    return '''
    document.querySelectorAll(".tab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll(".tab-btn").forEach(function (b) {
          b.classList.remove("active"); b.setAttribute("aria-selected", "false");
        });
        document.querySelectorAll(".tab-panel").forEach(function (p) { p.classList.remove("active"); });
        btn.classList.add("active"); btn.setAttribute("aria-selected", "true");
        var panel = document.getElementById("tab-" + btn.dataset.tab);
        if (panel) panel.classList.add("active");
        if (history.replaceState) history.replaceState(null, "", "#" + btn.dataset.tab);
      });
    });
    if (location.hash === "#detail") {
      var d = document.querySelector(\'[data-tab="detail"]\');
      if (d) d.click();
    }
    '''


def _waterfall_legend_html() -> str:
    items = [
        (STAGE_COLOR["s0_s1_input_query"], "S0+S1 Input & query"),
        (STAGE_COLOR["s2_retrieval"], "S2 Retrieval"),
        (STAGE_COLOR["s3_generation"], "S3 Generation"),
        (STAGE_COLOR["s4_rendering"], "S4 Rendering"),
    ]
    swatches = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:{hexc}"></span>{_esc(label)}</span>'
        for hexc, label in items
    )
    swatches += (
        '<span class="legend-item"><span class="legend-swatch" '
        'style="background:repeating-linear-gradient(45deg,var(--ink-muted) 0 1px,transparent 1px 4px)">'
        '</span>noise — indistinguishable from zero</span>'
        '<span class="legend-item"><span class="legend-swatch legend-swatch-void"></span>not measured</span>'
        f'<span class="legend-item"><span class="legend-swatch" style="border:2px solid {PALETTE["accent"]};'
        'background:transparent"></span>dominant stage</span>'
    )
    return f'<div class="waterfall-legend">{swatches}</div>'


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def render_html(report: Report, *, history: list[dict] | None = None) -> str:
    history = history or []
    current = {
        "generated_at": report.generated_at,
        "system_label": report.system_label,
        "languages": {lr.language: lr.quality.get("value") for lr in report.languages},
    }

    headline = build_headline(report)
    cards = language_cards(report)
    waterfall = render_waterfall_svg(report.cascades)

    overview = f'''
    <section id="tab-overview" class="tab-panel active" role="tabpanel">
      <div class="hero">
        <h1 class="headline">{_esc(headline)}</h1>
        {render_cards_html(cards)}
        <div class="waterfall-block">
          {waterfall}
          {_waterfall_legend_html()}
        </div>
      </div>
      {render_history_strip_html(history, current)}
      {trust_panel_html(report)}
    </section>'''

    detail = f'''
    <section id="tab-detail" class="tab-panel" role="tabpanel">
      <div class="detail-section">
        <h2>Per-stage metrics</h2>
        {detail_metrics_html(report)}
      </div>
      <div class="detail-section">
        <h2>Failure taxonomy</h2>
        {detail_taxonomy_html(report)}
      </div>
      <div class="detail-section">
        <h2>Calibration</h2>
        {detail_calibration_html(report)}
      </div>
      <div class="detail-section">
        <h2>Latency</h2>
        {detail_latency_html(report)}
      </div>
      <div class="detail-section">
        <h2>Run metadata</h2>
        {detail_metadata_html(report)}
      </div>
    </section>'''

    title = f"stratum · {report.system_label}"
    generated = report.generated_at

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="page">
  <header class="masthead">
    <div class="brand">
      <span class="brand-mark">STRATUM</span>
      <span class="brand-meta">{_esc(report.system_label)} · {_esc(generated)}</span>
    </div>
    <nav class="tabs" role="tablist">
      <button class="tab-btn active" data-tab="overview" role="tab" aria-selected="true">Overview</button>
      <button class="tab-btn" data-tab="detail" role="tab" aria-selected="false">Detail</button>
    </nav>
  </header>
  {overview}
  {detail}
  <footer class="page-footer">
    stratum report · {report.n_items} items · passes: {_esc(", ".join(report.passes_run))}
    {f' · dataset {_esc(report.dataset_hash)}' if report.dataset_hash else ''}
  </footer>
</div>
<script>{_js()}</script>
</body>
</html>'''


def write_html(report: Report, path: str | Path, *, history: list[dict] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report, history=history), encoding="utf-8")
    return path
