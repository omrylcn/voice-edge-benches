#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Pocket-TTS (Kyutai) C++ ONNX benchmark.

Unlike the other engines, Pocket has no Python/Rust ONNX binding here — it runs
through the VolgaGerm/PocketTTS.cpp C++ runtime. This wrapper drives that binary
as a subprocess, timing fp32 vs int8 on the same short/long sentences, and writes
result.json + WAVs under results/pocket-cpp/. The per-sentence schema matches the
other benches (p50_ms/p95_ms/mean_ms/rtf/wav_bytes); process-level fields
(cold_start_ms, rss_*) are absent because the engine runs out-of-process.

NOTE: because we shell out per synthesis, the measured time includes process
startup + model load each run. So Pocket's short-sentence numbers are inflated
vs the in-process (Python/Rust) benches — compare Pocket on the LONG sentence,
or read it as "end to end incl. load". The other engines measure in-process.
Also N_RUNS=3 (vs 10 elsewhere) because each run pays the full load cost; the
per-sentence "runs" field records this.

Build the binary first (see benches/pocket-cpp/README.md), then:
    POCKET_BIN=/path/to/pocket-tts python bench.py
Models are resolved from TTS_MODELS_DIR/pocket (default <repo>/models/pocket).
"""
from __future__ import annotations
import json
import os
import statistics
import struct
import subprocess
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("TTS_MODELS_DIR", REPO / "models")) / "pocket"
FIXTURES = REPO / "fixtures" / "sentences.json"
RESULTS_DIR = REPO / "results" / "pocket-cpp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BIN = os.environ.get("POCKET_BIN", "")
VOICE = os.environ.get("POCKET_VOICE", "piper-high")
VOICES_DIR = MODELS_DIR / "voices"
TOKENIZER = MODELS_DIR / "tokenizer.model"
N_RUNS = 3  # measured runs (after 1 warmup); keep low, subprocess is heavy


def wav_duration(path: str) -> float:
    """Read duration of a WAV (handles both PCM16 and IEEE-float fmt)."""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except wave.Error:
        pass
    # wave rejects IEEE-float (fmt tag 3); the binary writes float32 mono,
    # so frames = data bytes / 4.
    with open(path, "rb") as f:
        data = f.read()
    pos, sr, n = 12, 24000, 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        sz = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            sr = struct.unpack("<I", data[pos + 12:pos + 16])[0]
        elif cid == b"data":
            n = sz // 4
        pos += 8 + sz + (sz & 1)
    return n / sr


def run(precision: str, text: str, out: str) -> list[float]:
    cmd = [BIN, "--precision", precision, "--models-dir", str(MODELS_DIR),
           "--voices-dir", str(VOICES_DIR),
           "--tokenizer", str(TOKENIZER), text, VOICE, out]
    Path(out).unlink(missing_ok=True)  # never measure a stale WAV
    subprocess.run(cmd, capture_output=True)  # warmup (builds .emb cache)
    times = []
    for _ in range(N_RUNS):
        t = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        times.append((time.perf_counter() - t) * 1000)
        if proc.returncode != 0:
            raise SystemExit(f"pocket-tts failed (exit {proc.returncode}):\n"
                             f"{proc.stderr.strip()}")
    if not Path(out).exists():
        raise SystemExit(f"pocket-tts exited 0 but wrote no WAV: {out}")
    return times


def main() -> None:
    if not BIN or not Path(BIN).exists():
        raise SystemExit("Set POCKET_BIN to the built pocket-tts binary "
                         "(see benches/pocket-cpp/README.md).")
    fixtures = json.loads(FIXTURES.read_text())["sentences"]
    picks = {s["id"]: s for s in fixtures if s["id"] in ("short", "long")}

    for precision in ("fp32", "int8"):
        results = {"engine": "pocket", "runtime": f"cpp-{precision}",
                   "voice": VOICE, "sentences": []}
        for sid, s in picks.items():
            out = str(RESULTS_DIR / f"{precision}_{sid}.wav")
            times = run(precision, s["text"], out)
            times.sort()
            mean = statistics.mean(times)
            p50 = times[len(times) // 2]
            p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
            dur = wav_duration(out)
            # rtf from mean, matching the other benches
            rtf = (mean / 1000) / dur if dur else 0
            print(f"pocket {precision} {sid:5s}  mean {mean:8.1f}ms  "
                  f"p50 {p50:8.1f}ms  audio {dur:5.2f}s  RTF {rtf:.4f}")
            results["sentences"].append({
                "id": sid, "words": s["words"],
                "audio_duration_s": round(dur, 3), "sample_rate": 24000,
                "p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
                "mean_ms": round(mean, 1), "rtf": round(rtf, 4),
                "wav_bytes": Path(out).stat().st_size,
                "runs": N_RUNS,
            })
        out_path = RESULTS_DIR / f"result_{precision}.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
