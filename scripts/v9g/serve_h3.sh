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

exec sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant "$variant" \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --port "$port"
