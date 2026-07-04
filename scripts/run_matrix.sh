#!/usr/bin/env bash
# Thread-scaling matrix: run every bench pinned to N CPU cores and snapshot
# the result.json files under results/threads-<N>/.
#
# Mechanism: taskset pins the whole process to the first N cores — a uniform
# constraint that works even for engines whose ONNX session is buried inside a
# library. TTS_THREADS additionally caps ONNX intra-op threads where the bench
# supports it natively (supertonic-rust, pocket-cpp).
#
# Usage: scripts/run_matrix.sh [N ...]   (default: 1 4)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
THREAD_COUNTS=("${@:-1 4}")
[ $# -eq 0 ] && THREAD_COUNTS=(1 4)

cores_arg() { # N -> "0" or "0-3"
  local n
  n=$1
  if [ "$n" -le 1 ]; then echo "0"; else echo "0-$((n - 1))"; fi
}

snapshot() { # N: copy fresh result jsons into results/threads-N/, tagging thread count
  local n dest
  n=$1
  dest="$REPO/results/threads-$n"
  mkdir -p "$dest"
  for f in \
    results/piper-python-*/result.json \
    results/kokoro-python/result.json \
    results/piper-rust/result.json \
    results/supertonic-rust/result.json \
    results/pocket-cpp/result_fp32.json \
    results/pocket-cpp/result_int8.json; do
    [ -f "$REPO/$f" ] || continue
    local name
    name="$(echo "$f" | sed 's|results/||; s|/|_|g')"
    python3 - "$REPO/$f" "$dest/$name" "$n" <<'EOF'
import json, sys
data = json.loads(open(sys.argv[1]).read())
data["threads"] = int(sys.argv[3])
open(sys.argv[2], "w").write(json.dumps(data, indent=2))
EOF
  done
  echo "→ snapshot: $dest"
}

for N in ${THREAD_COUNTS[@]}; do
  CPUS="$(cores_arg "$N")"
  echo "===== threads=$N (taskset -c $CPUS) ====="
  export TTS_THREADS="$N" OMP_NUM_THREADS="$N"

  (cd "$REPO/benches/piper-python"  && taskset -c "$CPUS" uv run bench.py)
  (cd "$REPO/benches/kokoro-python" && taskset -c "$CPUS" uv run bench.py)
  (cd "$REPO/benches/piper-rust"    && taskset -c "$CPUS" ./target/release/piper-rust-bench 2>/dev/null \
     || (cd "$REPO/benches/piper-rust" && RUSTFLAGS="-l pcaudio" cargo run --release))
  (cd "$REPO/benches/supertonic-rust" && taskset -c "$CPUS" cargo run --release)
  (cd "$REPO/benches/pocket-cpp"    && POCKET_BIN=./pocket-tts taskset -c "$CPUS" python3 bench.py)

  snapshot "$N"
done

echo "Done. Compare results/threads-*/ side by side."
