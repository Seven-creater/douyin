#!/usr/bin/env bash
set -euo pipefail

# Official SGLang H3 contract: select the root model and a checkpoint variant.
# Start only one variant at a time during the first V9-G experiment so H3 and
# Omni do not compete for the same GPUs.
variant="${1:?usage: serve_h3.sh fl2va|ref2va [port]}"
port="${2:-30010}"
if [[ "$variant" != "fl2va" && "$variant" != "ref2va" ]]; then
  echo "variant must be fl2va or ref2va" >&2
  exit 2
fi

# sglang lives in the h3 conda env; detached launches (setsid/nohup) do not
# inherit it, so resolve the binary explicitly (overridable via SGLANG_BIN).
sglang_bin="${SGLANG_BIN:-/data02/usr/wangqihao/miniconda3/envs/h3/bin/sglang}"
if [[ ! -x "$sglang_bin" ]]; then
  echo "sglang binary not found: $sglang_bin" >&2
  exit 3
fi

exec "$sglang_bin" serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant "$variant" \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --port "$port"
