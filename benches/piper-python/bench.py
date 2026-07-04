# /// script
# requires-python = ">=3.10"
# dependencies = ["piper-tts>=1.2", "numpy", "psutil", "soundfile"]
# ///
"""Piper Python baseline benchmark.

Reuses the PiperVoice.synthesize_wav path. Runs each fixture sentence N times
warm (one warmup discarded), records wall-clock synthesis time, RAM delta, and
writes WAV output for ear-test.

Run with:  uv run bench.py          (uv reads the inline deps above — nothing to
install by hand). Or in a venv: pip install -e . && python bench.py
"""
from __future__ import annotations

import gc
import io
import json
import os
import statistics
import time
import tracemalloc
import wave
from pathlib import Path

import psutil

# Model dir is resolved via TTS_MODELS_DIR (default <repo>/models). Piper
# voices live under <models>/piper/<voice>.onnx.
REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("TTS_MODELS_DIR", REPO / "models"))
PIPER_MODEL_DIR = MODELS_DIR / "piper"
DEFAULT_VOICE = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
N_RUNS = 10  # 1 warmup + 9 measured
FIXTURES = REPO / "fixtures" / "sentences.json"
RESULTS_DIR = REPO / "results" / f"piper-python-{DEFAULT_VOICE}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def main() -> None:
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    fixtures = json.loads(FIXTURES.read_text())
    sentences = fixtures["sentences"]

    print(f"Loading Piper voice: {DEFAULT_VOICE}")
    rss_before_load = _rss_mb()
    t0 = time.perf_counter()
    model = PiperVoice.load(str(PIPER_MODEL_DIR / f"{DEFAULT_VOICE}.onnx"))
    cold_start_s = time.perf_counter() - t0
    rss_after_load = _rss_mb()

    print(f"  cold_start: {cold_start_s*1000:.0f} ms")
    print(f"  rss delta:  {rss_after_load - rss_before_load:.1f} MB")
    print()

    results = {
        "engine": "piper",
        "lang": "python",
        "voice": DEFAULT_VOICE,
        "cold_start_ms": round(cold_start_s * 1000, 1),
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_load_delta_mb": round(rss_after_load - rss_before_load, 1),
        "sentences": [],
    }

    syn_cfg = SynthesisConfig(length_scale=1.0)

    warmup_ms_total = 0.0
    for s in sentences:
        sid, text, words = s["id"], s["text"], s["words"]
        print(f"--- {sid} ({words} words) ---")
        timings_ms: list[float] = []
        ttfa_ms_runs: list[float] = []
        audio_bytes_last = b""

        # Warmup + measured. Chunked synthesize() (one chunk per sentence) so we
        # can stamp time-to-first-audio; chunks are assembled into the same WAV
        # synthesize_wav would have produced.
        for i in range(N_RUNS):
            gc.collect()
            t0 = time.perf_counter()
            ttfa_ms = 0.0
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                first = True
                for chunk in model.synthesize(text, syn_config=syn_cfg):
                    if first:
                        ttfa_ms = (time.perf_counter() - t0) * 1000
                        wf.setnchannels(chunk.sample_channels)
                        wf.setsampwidth(chunk.sample_width)
                        wf.setframerate(chunk.sample_rate)
                        first = False
                    wf.writeframes(chunk.audio_int16_bytes)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            audio_bytes_last = buf.getvalue()
            if i > 0:  # discard warmup
                timings_ms.append(elapsed_ms)
                ttfa_ms_runs.append(ttfa_ms)
            elif sid == sentences[0]["id"]:
                warmup_ms_total = elapsed_ms  # first-ever inference after load

        # Save audio output (last run)
        wav_path = RESULTS_DIR / f"{sid}.wav"
        wav_path.write_bytes(audio_bytes_last)

        # Audio metadata
        with wave.open(io.BytesIO(audio_bytes_last), "rb") as wf:
            sr = wf.getframerate()
            duration_s = wf.getnframes() / sr

        p50 = statistics.median(timings_ms)
        p95 = sorted(timings_ms)[int(len(timings_ms) * 0.95)] if len(timings_ms) > 1 else timings_ms[0]
        mean = statistics.mean(timings_ms)
        rtf = (mean / 1000) / duration_s  # real-time factor (<1 = faster than realtime)
        ttfa = statistics.mean(ttfa_ms_runs)

        print(f"  p50: {p50:.0f} ms | p95: {p95:.0f} ms | mean: {mean:.0f} ms | ttfa: {ttfa:.0f} ms")
        print(f"  audio: {duration_s:.2f}s @ {sr}Hz | RTF: {rtf:.3f}")
        print(f"  wav:   {wav_path.name} ({len(audio_bytes_last)} bytes)")

        results["sentences"].append({
            "id": sid,
            "words": words,
            "audio_duration_s": round(duration_s, 3),
            "sample_rate": sr,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "mean_ms": round(mean, 1),
            "ttfa_ms": round(ttfa, 1),
            "rtf": round(rtf, 4),
            "wav_bytes": len(audio_bytes_last),
            "runs": len(timings_ms),
        })

    rss_final = _rss_mb()
    results["warmup_ms"] = round(warmup_ms_total, 1)
    results["rss_final_mb"] = round(rss_final, 1)

    out_path = RESULTS_DIR / "result.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")
    print(f"RSS final: {rss_final:.1f} MB")


if __name__ == "__main__":
    main()
