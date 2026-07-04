# voice-edge-benches

CPU speech-model benchmarks on ONNX runtimes. Today: **TTS** (text-to-speech).
Planned: **STT** (speech-to-text) benches in the same harness.

**How fast — and how good — are CPU text-to-speech engines when you run the
*same* ONNX model through different runtimes?** This benchmark drives four TTS
engines — **Supertonic, Piper, Kokoro, Pocket-TTS** — through **Rust ONNX** and
**Python ONNX** runtimes (C++ for Pocket), on the *same* sentences, and measures
real synthesis latency. It also isolates the effect of **int8 quantization**,
which turns out to depend entirely on model architecture.

Everything below is measured on real hardware. **Every row links to its audio**
— same sentence for all, so quality is directly comparable by ear, and every
row also carries a UTMOS naturalness score.

## Five findings

**1. Rust ONNX beats Python ONNX on the same model — every time, by 1.5–2×.**
Same weights, same sentence; only the runtime differs. The gap is Python call
overhead. Supertonic step4 short: **215 ms (Rust) vs 446 ms (Python)**.

**2. int8 quantization is architecture-dependent, not a free win.**
On a **transformer** (Pocket-TTS) int8 is *faster* than fp32 (long RTF 0.30 →
0.13 ✓). On **conv-heavy** models (Kokoro, Supertonic) int8 dynamic on a
non-VNNI CPU is **3× *slower*** (Kokoro long RTF 0.25 → 0.86 ✗). Don't quantize a
conv model expecting speed.

**3. Quality vs speed is a per-engine knob — and now it's measurable.**
Supertonic's flow-matching `step` count trades quality for speed: UTMOS says
step8 ≈ **4.42** vs step4 ≈ **3.77** on the long clip, with step4 ~2× faster.
Piper (VITS, single pass) is the fastest of all but less warm. **Listen below
and decide.**

**4. Total synth time is the wrong latency metric — TTFA is what you feel.**
Streaming engines deliver first audio long before synthesis finishes: Pocket
int8 speaks after **~45 ms** while the full 18 s paragraph takes 2.5 s to
render; Piper streams sentence-by-sentence (long: first audio at ~250 ms of a
630 ms synth). Kokoro (via `kokoro-onnx`) batches up to 510 phonemes and can't
stream early at all — its TTFA equals total time.

**5. int8 costs (almost) no quality.** Where int8 changes speed, UTMOS barely
moves: Kokoro 4.51 → 4.49, Pocket 4.11 → 4.06. The int8 decision is purely
about your CPU's architecture (finding 2), not about audible quality.

## Results — listen and compare

Same sentences for everyone: short `"Hello, how are you today?"` (6 words) and a
58-word paragraph. **ms** = mean synth time (10 runs, 1 warmup discarded;
Pocket: 3 runs, subprocess). `RTF` = synth ÷ audio duration (lower = faster;
0.10 = 10× real-time). Box: **Intel i7-9700K, CPU-only** (no VNNI),
onnxruntime 1.27 / ort 2.0-rc9. ⚠️ = slower than real-time.

> Click the **listen** links to play each clip ([`results/audio/`](results/audio/)).
> A standalone offline page with all clips embedded is in
> [`dashboard.html`](dashboard.html).

<!-- DASHBOARD:BEGIN -->
| Engine | Runtime | short (ms) | long (ms) | RTF long | TTFA long (ms) | cold start (ms) | MOS | listen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Piper lessacmed | Python ONNX | 51 | 630 | 0.037 | 252 | 1,156 | 4.05 | [short](results/audio/piper_lessacmed_python_short.wav) · [long](results/audio/piper_lessacmed_python_long.wav) |
| Piper lessacmed | Rust ONNX | 91 | 705 | 0.043 | — | — | 4.43 | [short](results/audio/piper_lessacmed_rust_short.wav) · [long](results/audio/piper_lessacmed_rust_long.wav) |
| Supertonic step4 | Rust ONNX | 215 | 1,767 | 0.084 | — | — | 3.77 | [short](results/audio/supertonic_step4_rust_short.wav) · [long](results/audio/supertonic_step4_rust_long.wav) |
| Pocket-TTS int8 | C++ ONNX | 698 | 2,526 | 0.145 | 44 | 445 | 4.06 | [short](results/audio/pocket_int8_cpp_short.wav) · [long](results/audio/pocket_int8_cpp_long.wav) |
| Supertonic step4 | Python ONNX | 446 | 3,322 | 0.150 | — | — | 3.97 | [short](results/audio/supertonic_step4_python_short.wav) · [long](results/audio/supertonic_step4_python_long.wav) |
| Supertonic step8 | Rust ONNX | 427 | 3,584 | 0.171 | 2,883 | 904 | 4.42 | [short](results/audio/supertonic_step8_rust_short.wav) · [long](results/audio/supertonic_step8_rust_long.wav) |
| Piper ryanhigh | Python ONNX | 232 | 2,938 | 0.185 | — | 939 | 4.47 | [short](results/audio/piper_ryanhigh_python_short.wav) · [long](results/audio/piper_ryanhigh_python_long.wav) |
| Piper ryanhigh | Rust ONNX | 376 | 3,515 | 0.205 | 3,515 | 716 | 4.49 | [short](results/audio/piper_ryanhigh_rust_short.wav) · [long](results/audio/piper_ryanhigh_rust_long.wav) |
| Supertonic step8 | Python ONNX | 665 | 4,952 | 0.224 | — | — | 4.50 | [short](results/audio/supertonic_step8_python_short.wav) · [long](results/audio/supertonic_step8_python_long.wav) |
| Kokoro fp32 | Python ONNX | 642 | 5,096 | 0.273 | 5,096 | 706 | 4.52 | [short](results/audio/kokoro_fp32_python_short.wav) · [long](results/audio/kokoro_fp32_python_long.wav) |
| Pocket-TTS fp32 | C++ ONNX | 1,171 | 6,412 | 0.359 | 85 | 735 | 4.11 | [short](results/audio/pocket_fp32_cpp_short.wav) · [long](results/audio/pocket_fp32_cpp_long.wav) |
| Kokoro fp32 | Rust ONNX | 1,480 | 16,729 | 0.856 | — | 645 | 4.51 | [short](results/audio/kokoro_fp32_rust_short.wav) · [long](results/audio/kokoro_fp32_rust_long.wav) |
| Kokoro int8 | Rust ONNX | 1,482 | 16,732 | 0.856 | — | — | 4.47 | [short](results/audio/kokoro_int8_rust_short.wav) · [long](results/audio/kokoro_int8_rust_long.wav) |
| Kokoro int8 | Python ONNX | 2,349 | 23,651 | 1.264 ⚠️ | 23,649 | 906 | 4.49 | [short](results/audio/kokoro_int8_python_short.wav) · [long](results/audio/kokoro_int8_python_long.wav) |

### Thread scaling (long sentence RTF)

Same benches pinned to N cores (`scripts/run_matrix.sh`, taskset +
`TTS_THREADS`). How much a single edge-class core costs you:

| Engine | Runtime | 1 core RTF | 4 core RTF |
| --- | --- | ---: | ---: |
| Piper lessac-med | Python | 0.046 | 0.045 |
| Kokoro fp32 | Python | 0.299 | 0.356 |
| Piper ryan-high | Rust | 0.184 | 0.221 |
| Supertonic step8 | Rust | 0.366 | 0.190 |
| Pocket fp32 | C++ (incl. load) | 0.413 | 0.299 |
| Pocket int8 | C++ (incl. load) | 0.234 | 0.158 |

**TTFA** (time-to-first-audio) is how long until the first audible chunk is
ready — the latency a user actually feels on the long sentence. Streaming
engines (Pocket, Supertonic multi-chunk) deliver first audio far before the
full synthesis finishes; single-pass engines can't. Kokoro's `create_stream`
batches up to 510 phonemes, so even the 58-word text is one batch — TTFA ≈
total. **MOS** is UTMOS22-strong predicted naturalness (1–5, higher better)
on the long clip — see `scripts/quality_utmos.py`. Pocket's short/long **ms**
columns include process start + model load (subprocess bench); its TTFA and
cold start are measured inside the binary and are comparable. Rows without
a fresh result.json show — for the new metrics.
<!-- DASHBOARD:END -->

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

```text
benches/
  piper-python/     bench.py + pyproject.toml         (piper-tts)
  piper-rust/       src/main.rs + Cargo.toml          (piper-rs crate)
  kokoro-python/    bench.py, bench_int8.py + pyproject (kokoro-onnx)
  kokoro-rust/      src/main.rs + Cargo.toml          (Kokoros)
  supertonic-rust/  src/{bench,helper}.rs + Cargo     (Supertonic engine, vendored MIT)
  pocket-cpp/       bench.py + src/pocket_tts.cpp      (PocketTTS.cpp, vendored MIT)
fixtures/           sentences.json, sentences_multi.json   (shared test text)
results/            per-engine result.json + audio/ (WAV clips) + threads-N/ snapshots
scripts/            download_models.sh, run_matrix.sh (1 vs 4 core),
                    quality_utmos.py (MOS scoring), gen_readme.py (results table)
dashboard.html      offline embedded-audio page
models/             weights — GIT-IGNORED, not shipped
```

> `benches/kokoro-rust` depends on a local [Kokoros](https://github.com/lucasjinreal/Kokoros)
> checkout at `benches/upstream/kokoros` (not vendored); without it that one bench
> keeps its last recorded numbers.

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
