# voice-onnx-benches

CPU speech-model benchmarks on ONNX runtimes. Today: **TTS** (text-to-speech).
Planned: **STT** (speech-to-text) benches in the same harness.

**How fast — and how good — are CPU text-to-speech engines when you run the
*same* ONNX model through different runtimes?** This benchmark drives four TTS
engines — **Supertonic, Piper, Kokoro, Pocket-TTS** — through **Rust ONNX** and
**Python ONNX** runtimes (C++ for Pocket), on the *same* sentences, and measures
real synthesis latency. It also isolates the effect of **int8 quantization**,
which turns out to depend entirely on model architecture.

Everything below is measured on real hardware. **Every row has audio you can play
right here in the README** — same sentence for all, so quality is directly
comparable by ear.

## Three findings

**1. Rust ONNX beats Python ONNX on the same model — every time, by 1.5–2×.**
Same weights, same sentence; only the runtime differs. The gap is Python call
overhead. Supertonic step4 short: **215 ms (Rust) vs 446 ms (Python)**.

**2. int8 quantization is architecture-dependent, not a free win.**
On a **transformer** (Pocket-TTS) int8 is *faster* than fp32 (long RTF 0.30 →
0.13 ✓). On **conv-heavy** models (Kokoro, Supertonic) int8 dynamic on a
non-VNNI CPU is **3× *slower*** (Kokoro long RTF 0.25 → 0.86 ✗). Don't quantize a
conv model expecting speed.

**3. Quality vs speed is a per-engine knob.** Supertonic's flow-matching `step`
count trades quality for speed: step8 is smoothest but slowest, step4 is nearly
indistinguishable by ear and ~2× faster. Piper (VITS, single pass) is the fastest
of all but less warm. **Listen below and decide.**

## Results — listen and compare

Same sentences for everyone: short `"Hello, how are you today?"` (6 words) and a
58-word paragraph. `p50` = median synth time over 3 runs. `RTF` = synth ÷ audio
duration (lower = faster; 0.10 = 10× real-time). Box: **Intel i7-9700K, CPU-only**
(no VNNI), onnxruntime 1.27 / ort 2.0-rc9. ⚠️ = slower than real-time.

> Audio players render on GitHub. If your viewer strips `<audio>`, the WAVs are in
> [`results/audio/`](results/audio/). A standalone offline page with all clips is
> in [`dashboard.html`](dashboard.html).

<!-- DASHBOARD -->

### Reading the numbers

- **Runtime, not the model, sets the speed floor.** Every engine is faster in
  Rust ONNX than Python ONNX — identical weights, so it's pure runtime overhead.
- **int8's ⚠️ rows** (Kokoro) are the headline caveat: dynamic int8 on conv
  layers without VNNI is a regression, not an optimization. Pocket (transformer)
  is the opposite — int8 is its fastest mode.
- **Pocket's short-sentence times are inflated**: its bench shells out per
  synthesis, so process start + model load are included. Compare Pocket on the
  **long** sentence. Every other engine measures in-process.
- **Best overall balance: Supertonic step4 on Rust ONNX** — near-step8 quality,
  ~2× faster, warm 44.1 kHz. **Fastest raw: Piper lessac-medium** (long RTF
  0.04) but a less natural voice.

## Repo layout

```
benches/
  piper-python/     bench.py + pyproject.toml         (piper-tts)
  piper-rust/       src/main.rs + Cargo.toml          (piper-rs crate)
  kokoro-python/    bench.py, bench_int8.py + pyproject (kokoro-onnx)
  kokoro-rust/      src/main.rs + Cargo.toml          (Kokoros)
  supertonic-rust/  src/{bench,helper}.rs + Cargo     (Supertonic engine, vendored MIT)
  pocket-cpp/       bench.py + src/pocket_tts.cpp      (PocketTTS.cpp, vendored MIT)
fixtures/           sentences.json, sentences_multi.json   (shared test text)
results/            per-engine result.json + audio/ (WAV clips)
scripts/            download_models.sh
dashboard.html      offline embedded-audio page
models/             weights — GIT-IGNORED, not shipped
```

## Run it yourself

### 1. Get the models (into `models/`, git-ignored)

```bash
scripts/download_models.sh all          # piper + kokoro
scripts/download_models.sh supertonic   # OpenRAIL-M — review license first
scripts/download_models.sh pocket       # export instructions
```

Or point at an existing copy with `TTS_MODELS_DIR=/path/to/models`. Expected
layout: `models/piper/*.onnx(.json)`, `models/kokoro-v1.0*.onnx` +
`voices-v1.0.bin`, `models/supertonic/{onnx,voice_styles}`, `models/pocket/*.onnx`.

### 2. Run a bench

**Python** (each dir has a `pyproject.toml` — `uv run bench.py`, or
`pip install -e .` then `python bench.py`):

```bash
cd benches/piper-python   && uv run bench.py     # PIPER_VOICE=en_US-ryan-high to switch
cd benches/kokoro-python  && uv run bench.py     # + uv run bench_int8.py (slow, non-VNNI)
cd benches/pocket-cpp     && POCKET_BIN=/path/to/pocket-tts uv run bench.py
```

**Rust**:

```bash
cargo run --release --manifest-path benches/piper-rust/Cargo.toml
cargo run --release --manifest-path benches/kokoro-rust/Cargo.toml
cargo run --release --bin bench --manifest-path benches/supertonic-rust/Cargo.toml
```

Each writes `result.json` + WAVs under `results/`. `TTS_MODELS_DIR` overrides
where models are read from.

> **Linux/Piper Rust:** the build links espeak-ng, which needs `libpcaudio`.
> Install `libpcaudio-dev` or build with `RUSTFLAGS="-l pcaudio" cargo build`.

## Licenses

The **benchmark harness** (`benches/`, `fixtures/`, `scripts/`) is MIT — see
[LICENSE](LICENSE). The **engines and model weights** are third-party with their
own licenses — see [NOTICE](NOTICE). Vendored engine code (`supertonic-rust`
helper, `pocket-cpp` source) keeps its upstream MIT license with attribution.
The Supertonic weights are **BigScience OpenRAIL-M** (use-based restrictions) and
are **not redistributed** here.
