# Pocket-TTS (Kyutai) — C++ ONNX bench

Pocket-TTS is autoregressive (transformer LM + Mimi codec). It has no Rust/Python
ONNX binding in this repo — it runs through the C++ runtime
[VolgaGerm/PocketTTS.cpp](https://github.com/VolgaGerm/PocketTTS.cpp) (MIT). We
vendor the runtime source here (`src/pocket_tts.cpp`, `CMakeLists.txt`,
`export_onnx.py`) for attribution and reproducibility; see `LICENSE.upstream`.

## Why it's the odd one out

Our `bench.py` shells out to the compiled binary once per synthesis, so its
timing includes **process start + model load each run**. The Python/Rust benches
measure in-process (model stays loaded). ⇒ Compare Pocket on the **long**
sentence, or read its numbers as "end to end including load". It's the honest
apples-to-oranges caveat — don't compare Pocket's short-sentence ms to the others'.

## Build

```bash
# 1. Get the ONNX models (into models/pocket/, git-ignored)
../../scripts/download_models.sh pocket

# 2. Build the C++ runtime (needs cmake + onnxruntime headers/libs)
git clone https://github.com/VolgaGerm/PocketTTS.cpp /tmp/PocketTTS.cpp
cd /tmp/PocketTTS.cpp && cmake -B build && cmake --build build
# → produces the `pocket-tts` binary
```

(The vendored `src/pocket_tts.cpp` here matches that repo; you can also build it
directly with the upstream's CMakeLists.)

## Run

```bash
POCKET_BIN=/tmp/PocketTTS.cpp/pocket-tts python bench.py
```

Writes `result_fp32.json`, `result_int8.json` + WAVs under
`results/pocket-cpp/`. `TTS_MODELS_DIR` overrides the model dir;
`POCKET_VOICE` (default `piper-high`) picks the voice sample.

## Finding

int8 **helps** here — Pocket is a transformer, so int8 is faster than fp32
(long RTF 0.30 → 0.13). Contrast Kokoro/Supertonic (conv), where int8 on a
non-VNNI CPU is *slower*. That architecture split is the whole point of the
benchmark.
