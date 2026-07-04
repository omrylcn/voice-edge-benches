# /// script
# requires-python = ">=3.10"
# dependencies = ["kokoro-onnx>=0.4", "numpy", "psutil", "soundfile"]
# ///
"""Kokoro Python baseline benchmark using kokoro-onnx package.

Run with:  uv run bench.py   (uv reads the inline deps above).

Mirrors piper-python/bench.py: same fixtures, same N_RUNS protocol, same
output schema. Uses af_heart voice (Kokoro's default).

Run with: ./.venv/bin/python bench.py
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import time
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
from kokoro_onnx import Kokoro

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("TTS_MODELS_DIR", REPO / "models"))
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"
VOICE = "af_heart"
N_RUNS = 10  # 1 warmup + 9 measured
FIXTURES = REPO / "fixtures" / "sentences.json"
RESULTS_DIR = REPO / "results" / "kokoro-python"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _samples_to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = BytesIO()
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def main() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    sentences = fixtures["sentences"]

    print(f"Loading Kokoro voice: {VOICE}")
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    cold_start_s = time.perf_counter() - t0
    rss_after = _rss_mb()

    print(f"  cold_start: {cold_start_s*1000:.0f} ms")
    print(f"  rss delta:  {rss_after - rss_before:.1f} MB")
    print()

    results = {
        "engine": "kokoro",
        "lang": "python",
        "voice": VOICE,
        "cold_start_ms": round(cold_start_s * 1000, 1),
        "rss_after_load_mb": round(rss_after, 1),
        "rss_load_delta_mb": round(rss_after - rss_before, 1),
        "sentences": [],
    }

    for s in sentences:
        sid, text, words = s["id"], s["text"], s["words"]
        print(f"--- {sid} ({words} words) ---")
        timings_ms: list[float] = []
        last_audio: np.ndarray = np.zeros(0)
        last_sr: int = 24000

        for i in range(N_RUNS):
            gc.collect()
            t0 = time.perf_counter()
            audio, sr = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if i > 0:
                timings_ms.append(elapsed_ms)
            last_audio = audio
            last_sr = sr

        wav_bytes = _samples_to_wav_bytes(last_audio, last_sr)
        wav_path = RESULTS_DIR / f"{sid}.wav"
        wav_path.write_bytes(wav_bytes)

        duration_s = len(last_audio) / last_sr
        p50 = statistics.median(timings_ms)
        p95 = sorted(timings_ms)[int(len(timings_ms) * 0.95)] if len(timings_ms) > 1 else timings_ms[0]
        mean = statistics.mean(timings_ms)
        rtf = (mean / 1000) / duration_s

        print(f"  p50: {p50:.0f} ms | p95: {p95:.0f} ms | mean: {mean:.0f} ms")
        print(f"  audio: {duration_s:.2f}s @ {last_sr}Hz | RTF: {rtf:.3f}")
        print(f"  wav:   {wav_path.name} ({len(wav_bytes)} bytes)")

        results["sentences"].append({
            "id": sid,
            "words": words,
            "audio_duration_s": round(duration_s, 3),
            "sample_rate": last_sr,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "mean_ms": round(mean, 1),
            "rtf": round(rtf, 4),
            "wav_bytes": len(wav_bytes),
            "runs": len(timings_ms),
        })

    rss_final = _rss_mb()
    results["rss_final_mb"] = round(rss_final, 1)
    out_path = RESULTS_DIR / "result.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")
    print(f"RSS final: {rss_final:.1f} MB")


if __name__ == "__main__":
    main()
