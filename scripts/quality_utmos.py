# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "soundfile", "numpy"]
# ///
"""UTMOS quality scoring for the benchmark WAVs.

Runs the UTMOS22-strong MOS predictor (torch.hub tarepan/SpeechMOS, MIT) over
every WAV in results/audio/ and writes results/quality_utmos.json mapping
filename -> predicted MOS (1-5, higher = more natural).

Usage:  uv run scripts/quality_utmos.py
Needs network on first run (torch.hub model download, ~400MB torch + weights).
CPU-only is fine; ~1s per clip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO / "results" / "audio"
OUT_PATH = REPO / "results" / "quality_utmos.json"


def main() -> None:
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No WAVs in {AUDIO_DIR} — run the benches first.")

    print("Loading UTMOS22-strong (torch.hub tarepan/SpeechMOS)…")
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True)
    predictor.eval()

    scores: dict[str, float] = {}
    for wav in wavs:
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        with torch.no_grad():
            mos = predictor(torch.from_numpy(np.ascontiguousarray(audio))[None, :], sr)
        scores[wav.name] = round(float(mos.item()), 3)
        print(f"  {wav.name:45s} MOS {scores[wav.name]:.3f}")

    OUT_PATH.write_text(json.dumps(scores, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
