# tts-onnx-bench

**How fast — and how good — are CPU TTS engines when you run the *same* ONNX
model through different runtimes?** This benchmark drives four text-to-speech
engines (Supertonic, Piper, Kokoro, Pocket-TTS) through **Rust ONNX** and
**Python ONNX** runtimes (plus C++ for Pocket), on the *same* sentences, and
measures real synthesis latency. It also isolates the effect of **int8
quantization** — which turns out to depend entirely on model architecture.

All numbers below are measured, not estimated. Every row produced a real WAV you
can listen to (see `dashboard.html` / `results/`).

## The three findings

1. **Rust ONNX beats Python ONNX on the same model, every time** — by 1.5–2×.
   Same weights, same sentence, only the runtime differs; the gap is Python
   call overhead. E.g. Supertonic step4 short: **215 ms (Rust) vs 446 ms
   (Python)**.

2. **int8 quantization is architecture-dependent, not a free win.**
   - On a **transformer** (Pocket-TTS): int8 is *faster* than fp32 (long RTF
     0.30 → 0.13). ✓
   - On **conv-heavy** models (Kokoro, Supertonic): int8 dynamic on a non-VNNI
     CPU is **3× *slower*** than fp32 (Kokoro long RTF 0.25 → 0.86). ✗
   Don't quantize a conv model and expect speed.

3. **Quality vs speed is a per-engine knob.** Supertonic's `step` count
   (flow-matching denoising steps) trades quality for speed: step8 is smoothest
   but slowest; step4 is nearly indistinguishable by ear and ~2× faster. Piper
   (VITS, single pass) is the fastest of all but less warm.

## Results

Same sentences for everyone — short `"Hello, how are you today?"` (6 words) and
a 58-word paragraph. `p50` = median synth time over 3 runs. `RTF` = synth time ÷
audio duration (lower = faster; 0.10 means 10× real-time). Box: Intel i7-9700K,
CPU-only, onnxruntime 1.27 / ort 2.0-rc9. ⚠ = slower than real-time.

| Engine | Variant | Runtime | short p50 | short RTF | long p50 | long RTF |
|---|---|---|---:|---:|---:|---:|
| Supertonic | step8 | Rust ONNX | 403 ms | 0.220 | 3210 ms | 0.154 |
| Supertonic | step8 | Python ONNX | 665 ms | 0.341 | 4952 ms | 0.224 |
| Supertonic | step4 | Rust ONNX | 215 ms | 0.118 | 1767 ms | 0.084 |
| Supertonic | step4 | Python ONNX | 446 ms | 0.229 | 3322 ms | 0.150 |
| Piper | ryan-high | Rust ONNX | 232 ms | 0.178 | 2802 ms | 0.161 |
| Piper | ryan-high | Python ONNX | 191 ms | 0.168 | 2618 ms | 0.185 |
| Piper | lessac-medium | Rust ONNX | 91 ms | 0.059 | 705 ms | 0.043 |
| Piper | lessac-medium | Python ONNX | 54 ms | 0.037 | 657 ms | 0.040 |
| Kokoro | fp32 | Rust ONNX | 440 ms | 0.213 | 4879 ms | 0.250 |
| Kokoro | fp32 | Python ONNX | 557 ms | 0.408 | 5944 ms | 0.318 |
| Kokoro | int8 | Rust ONNX | 1482 ms | 0.705 | 16732 ms | 0.856 ⚠ |
| Kokoro | int8 | Python ONNX | 2226 ms | 1.490 | 22459 ms | 1.205 ⚠ |
| Pocket-TTS | fp32 | C++ ONNX | 1142 ms | 0.951 | 5266 ms | 0.301 |
| Pocket-TTS | int8 | C++ ONNX | 668 ms | 0.380 | 2414 ms | 0.134 |

Listen to all 28 clips in **`dashboard.html`** (open in a browser — audio is
embedded).

## Repo layout

```
benches/
  piper-python/     bench.py            (piper-tts)
  piper-rust/       src/main.rs         (piper-rs crate)
  kokoro-python/    bench.py, bench_int8.py   (kokoro-onnx)
  kokoro-rust/      src/main.rs         (Kokoros)
  supertonic-rust/  src/{bench,helper}.rs   (Supertonic engine, vendored MIT)
fixtures/           sentences.json, sentences_multi.json  (the shared test text)
results/            per-engine result.json + sample WAVs
scripts/            download_models.sh
dashboard.html      embedded-audio comparison page
models/             weights — GIT-IGNORED, not shipped (see below)
```

## Run it yourself

### 1. Get the models

Model weights are **not** committed (large + license-restricted). Put them under
`models/` (git-ignored). Either fetch them:

```bash
scripts/download_models.sh all          # piper + kokoro
scripts/download_models.sh supertonic   # OpenRAIL-M — review license first
```

…or point at an existing copy with `TTS_MODELS_DIR=/path/to/models`. Expected
layout:

```
models/
  piper/en_US-lessac-medium.onnx(.json), en_US-ryan-high.onnx(.json)
  kokoro-v1.0.onnx, kokoro-v1.0.int8.onnx, voices-v1.0.bin
  supertonic/onnx/*.onnx (+ tts.json, unicode_indexer.json)
  supertonic/voice_styles/F1.json
```

### 2. Run a bench

Python (each dir has its own venv needs — `pip install piper-tts` /
`kokoro-onnx`, plus `psutil soundfile numpy`):

```bash
python benches/piper-python/bench.py            # PIPER_VOICE=en_US-ryan-high to switch
python benches/kokoro-python/bench.py
python benches/kokoro-python/bench_int8.py       # slow on non-VNNI CPUs
```

Rust:

```bash
cargo run --release --manifest-path benches/piper-rust/Cargo.toml
cargo run --release --manifest-path benches/kokoro-rust/Cargo.toml
cargo run --release --bin bench --manifest-path benches/supertonic-rust/Cargo.toml
```

Each writes `result.json` + WAVs under `results/`. Set `TTS_MODELS_DIR` to
override where models are read from.

> Note (Linux): the Piper Rust build links espeak-ng, which needs `libpcaudio`.
> Install `libpcaudio-dev`, or build with
> `RUSTFLAGS="-l pcaudio" cargo build --release`.

## Licenses

The **benchmark harness** (everything in `benches/`, `fixtures/`, `scripts/`) is
MIT — see [LICENSE](LICENSE). The **engines and model weights** are third-party
and keep their own licenses — see [NOTICE](NOTICE) for full attribution. Notably
the Supertonic weights are **BigScience OpenRAIL-M** (use-based restrictions) and
are **not redistributed** here.
