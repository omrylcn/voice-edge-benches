#!/usr/bin/env python3
"""Regenerate the README results section from results/*.json.

Row skeleton comes from results/summary_14rows.json (the curated 14
configurations). Where a per-engine result.json exists, its fresh numbers
(mean_ms, rtf, ttfa_ms, cold_start_ms) override the skeleton; rows without a
fresh file keep the skeleton numbers and show "—" for the new metrics.
MOS columns come from results/quality_utmos.json (UTMOS22-strong).

Output replaces everything between <!-- DASHBOARD:BEGIN --> and
<!-- DASHBOARD:END --> in README.md (a bare <!-- DASHBOARD --> marker is
upgraded to the pair on first run).

Usage: python3 scripts/gen_readme.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
README = REPO / "README.md"

# row key in summary_14rows.json -> (result file, short id, long id)
FRESH = {
    "piper|lessacmed|python": ("piper-python-en_US-lessac-medium/result.json", "short", "long"),
    "piper|ryanhigh|python": ("piper-python-en_US-ryan-high/result.json", "short", "long"),
    "piper|ryanhigh|rust": ("piper-rust/result.json", "short", "long"),
    "piper|lessacmed|rust": ("piper-rust-en_US-lessac-medium/result.json", "short", "long"),
    "kokoro|fp32|python": ("kokoro-python/result.json", "short", "long"),
    "kokoro|int8|python": ("kokoro-python-int8/result.json", "short", "long"),
    "kokoro|fp32|rust": ("kokoro-rust/result.json", "short", "long"),
    "kokoro|int8|rust": ("kokoro-rust-int8/result.json", "short", "long"),
    "supertonic|step8|rust": ("supertonic-rust/result.json", "en_short", "en_long"),
    "supertonic|step4|rust": ("supertonic-rust-step4/result.json", "en_short", "en_long"),
    "pocket|fp32|cpp": ("pocket-cpp/result_fp32.json", "short", "long"),
    "pocket|int8|cpp": ("pocket-cpp/result_int8.json", "short", "long"),
}


def _fmt_ms(v) -> str:
    return f"{v:,.0f}" if isinstance(v, (int, float)) else "—"


def _sentence(data: dict, sid: str) -> dict:
    for s in data.get("sentences", []):
        if s["id"] == sid:
            return s
    return {}


def build_rows() -> list[dict]:
    summary = json.loads((RESULTS / "summary_14rows.json").read_text())
    mos = {}
    mos_path = RESULTS / "quality_utmos.json"
    if mos_path.exists():
        mos = json.loads(mos_path.read_text())

    rows = []
    for key, base in summary.items():
        engine, variant, runtime = key.split("|")
        row = {
            "engine": base["engine"],
            "variant": variant,
            "runtime": base["runtime"],
            "short_ms": base.get("short_ms"),
            "long_ms": base.get("long_ms"),
            "long_rtf": base.get("long_rtf"),
            "ttfa_ms": None,
            "cold_ms": None,
        }
        if key in FRESH:
            path, sid, lid = FRESH[key]
            f = RESULTS / path
            if f.exists():
                data = json.loads(f.read_text())
                sh, lg = _sentence(data, sid), _sentence(data, lid)
                if sh.get("mean_ms") is not None:
                    row["short_ms"] = sh["mean_ms"]
                if lg.get("mean_ms") is not None:
                    row["long_ms"] = lg["mean_ms"]
                if lg.get("rtf") is not None:
                    row["long_rtf"] = lg["rtf"]
                row["ttfa_ms"] = lg.get("ttfa_ms")
                row["cold_ms"] = data.get("cold_start_ms")

        audio_base = f"{engine}_{variant}_{runtime}"
        wav_short = f"{audio_base}_short.wav"
        wav_long = f"{audio_base}_long.wav"
        row["wav_short"] = wav_short if (RESULTS / "audio" / wav_short).exists() else None
        row["wav_long"] = wav_long if (RESULTS / "audio" / wav_long).exists() else None
        row["mos_long"] = mos.get(wav_long)
        rows.append(row)

    rows.sort(key=lambda r: (r["long_rtf"] is None, r["long_rtf"] or 0))
    return rows


def render_threads() -> str:
    """Small 1-core vs 4-core RTF table from results/threads-N/ snapshots."""
    snaps: dict[int, dict[str, dict]] = {}
    for d in sorted(RESULTS.glob("threads-*")):
        try:
            n = int(d.name.split("-")[1])
        except ValueError:
            continue
        snaps[n] = {f.name: json.loads(f.read_text()) for f in d.glob("*.json")}
    if len(snaps) < 2:
        return ""

    counts = sorted(snaps)
    label = {
        "piper-python-en_US-lessac-medium_result.json": ("Piper lessac-med", "Python", "long"),
        "kokoro-python_result.json": ("Kokoro fp32", "Python", "long"),
        "piper-rust_result.json": ("Piper ryan-high", "Rust", "long"),
        "supertonic-rust_result.json": ("Supertonic step8", "Rust", "en_long"),
        "pocket-cpp_result_fp32.json": ("Pocket fp32", "C++ (incl. load)", "long"),
        "pocket-cpp_result_int8.json": ("Pocket int8", "C++ (incl. load)", "long"),
    }
    lines = [
        "### Thread scaling (long sentence RTF)",
        "",
        "Same benches pinned to N cores (`scripts/run_matrix.sh`, taskset +",
        "`TTS_THREADS`). How much a single edge-class core costs you:",
        "",
        "| Engine | Runtime | " + " | ".join(f"{n} core RTF" for n in counts) + " |",
        "| --- | --- |" + " ---: |" * len(counts),
    ]
    for fname, (eng, rt, lid) in label.items():
        vals = []
        for n in counts:
            data = snaps[n].get(fname)
            s = _sentence(data, lid) if data else {}
            vals.append(f"{s['rtf']:.3f}" if s.get("rtf") is not None else "—")
        if any(v != "—" for v in vals):
            lines.append(f"| {eng} | {rt} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render(rows: list[dict]) -> str:
    out = [
        "| Engine | Runtime | short (ms) | long (ms) | RTF long | TTFA long (ms) | cold start (ms) | MOS | listen |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        warn = " ⚠️" if (r["long_rtf"] or 0) > 1 else ""
        links = []
        if r["wav_short"]:
            links.append(f"[short](results/audio/{r['wav_short']})")
        if r["wav_long"]:
            links.append(f"[long](results/audio/{r['wav_long']})")
        rtf = f"{r['long_rtf']:.3f}" if r["long_rtf"] is not None else "—"
        mos_v = f"{r['mos_long']:.2f}" if r["mos_long"] is not None else "—"
        out.append(
            f"| {r['engine']} {r['variant']} | {r['runtime']} "
            f"| {_fmt_ms(r['short_ms'])} | {_fmt_ms(r['long_ms'])} | {rtf}{warn} "
            f"| {_fmt_ms(r['ttfa_ms'])} | {_fmt_ms(r['cold_ms'])} | {mos_v} "
            f"| {' · '.join(links) if links else '—'} |"
        )
    threads_block = render_threads()
    if threads_block:
        out += ["", threads_block]
    out += [
        "",
        "**TTFA** (time-to-first-audio) is how long until the first audible chunk is",
        "ready — the latency a user actually feels on the long sentence. Streaming",
        "engines (Pocket, Supertonic multi-chunk) deliver first audio far before the",
        "full synthesis finishes; single-pass engines can't. Kokoro's `create_stream`",
        "batches up to 510 phonemes, so even the 58-word text is one batch — TTFA ≈",
        "total. **MOS** is UTMOS22-strong predicted naturalness (1–5, higher better)",
        "on the long clip — see `scripts/quality_utmos.py`. Pocket's short/long **ms**",
        "columns include process start + model load (subprocess bench); its TTFA and",
        "cold start are measured inside the binary and are comparable. Rows without",
        "a fresh result.json show — for the new metrics.",
    ]
    return "\n".join(out)


def main() -> None:
    text = README.read_text()
    if "<!-- DASHBOARD:BEGIN -->" not in text:
        text = text.replace(
            "<!-- DASHBOARD -->",
            "<!-- DASHBOARD:BEGIN -->\n<!-- DASHBOARD:END -->",
        )
    block = f"<!-- DASHBOARD:BEGIN -->\n{render(build_rows())}\n<!-- DASHBOARD:END -->"
    text = re.sub(
        r"<!-- DASHBOARD:BEGIN -->.*?<!-- DASHBOARD:END -->",
        lambda _: block,
        text,
        flags=re.S,
    )
    README.write_text(text)
    print(f"README results section regenerated ({len(build_rows())} rows).")


if __name__ == "__main__":
    main()
