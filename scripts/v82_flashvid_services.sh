#!/usr/bin/env bash
set -euo pipefail

# V8.2 uses two GPUs for recall browsing, then releases both before the four
# dual-GPU Omni workers start. Only process groups carrying our ownership
# marker and PID files are stopped by this script.

DOUYIN_DIR="${V82_DOUYIN_DIR:-/data02/usr/wangqihao/Demo/research}"
FLASHVID_DIR="${V82_FLASHVID_DIR:-/data02/usr/wangqihao/Demo/test/flashvid_v82_audit}"
MODEL_PATH="${V82_MODEL_PATH:-/data02/usr/wangqihao/Demo/test/flashvid/models/Qwen3.5-4B}"
PYTHON_BIN="${V82_VLLM_PYTHON:-/data02/usr/wangqihao/Demo/test/flashvid/.venv/bin/python}"
MEDIA_ROOT="${V82_MEDIA_ROOT:-/data02/usr/wangqihao/Demo}"
STATE_DIR="${V82_STATE_DIR:-${DOUYIN_DIR}/data/agentic_runs/.v82_browse_services}"
mkdir -p "$STATE_DIR"

public_port() { case "$1" in r025) echo 8102 ;; r100) echo 8104 ;; *) return 2 ;; esac; }
upstream_port() { case "$1" in r025) echo 18102 ;; r100) echo 18104 ;; *) return 2 ;; esac; }
ratio() { case "$1" in r025) echo 0.25 ;; r100) echo 1.00 ;; *) return 2 ;; esac; }
backend() { case "$1" in r025) echo flashvid ;; r100) echo native_bypass ;; *) return 2 ;; esac; }
gpu() { case "$1" in r025) echo "${V82_GPU_R025:-0}" ;; r100) echo "${V82_GPU_R100:-1}" ;; *) return 2 ;; esac; }

owned() {
  local marker="$1" pid="$2"
  [[ -r "/proc/$pid/environ" ]] || return 1
  tr '\0' '\n' < "/proc/$pid/environ" | grep -Fxq "DOUYIN_V82_SERVICE_ID=$marker"
}

wait_http() {
  local url="$1"
  for _ in {1..180}; do
    if "$PYTHON_BIN" - "$url" <<'PY' >/dev/null 2>&1
import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

stop_pid() {
  local marker="$1" pidfile="$2" pid
  [[ -f "$pidfile" ]] || return 0
  pid="$(tr -d '[:space:]' < "$pidfile")"
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && owned "$marker" "$pid"; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
  rm -f -- "$pidfile"
}

stop_one() {
  local id="$1"
  stop_pid "${id}_proxy" "$STATE_DIR/${id}_proxy.pid"
  stop_pid "${id}_upstream" "$STATE_DIR/${id}_upstream.pid"
}

start_one() {
  local id="$1" pub up keep service_backend service_gpu upstream_log proxy_log audit_file
  pub="$(public_port "$id")"; up="$(upstream_port "$id")"; keep="$(ratio "$id")"
  service_backend="$(backend "$id")"; service_gpu="$(gpu "$id")"
  upstream_log="$STATE_DIR/${id}_upstream.log"; proxy_log="$STATE_DIR/${id}_proxy.log"
  audit_file="$STATE_DIR/${id}_model_audit.jsonl"
  stop_one "$id"
  : > "$audit_file"
  common_args=(--default-chat-template-kwargs '{"enable_thinking":false}'
    --host 127.0.0.1 --port "$up" --tensor-parallel-size 1 --dtype bfloat16
    --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 32768
    --gpu-memory-utilization 0.90 --enable-prompt-tokens-details
    --limit-mm-per-prompt '{"image":4,"video":1}'
    --allowed-local-media-path "$MEDIA_ROOT")
  if [[ "$id" == r025 ]]; then
    startup=("$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_PATH"
      --hf-overrides '{"architectures":["FlashVIDQwen3_5ForConditionalGeneration"]}'
      --video-pruning-rate 0.75 --served-model-name Qwen3.5-4B-FlashVID-r025
      "${common_args[@]}")
    setsid nohup env DOUYIN_V82_SERVICE_ID="${id}_upstream" CUDA_VISIBLE_DEVICES="$service_gpu" \
      VLLM_PLUGINS=flashvid_qwen3_5 FLASHVID_VISION_RETENTION_RATIO="$keep" \
      FLASHVID_AUDIT_JSONL="$audit_file" PYTHONPATH="$FLASHVID_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
      "${startup[@]}" >"$upstream_log" 2>&1 < /dev/null &
  else
    startup=("$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_PATH"
      --served-model-name Qwen3.5-4B-FlashVID-r100 "${common_args[@]}")
    setsid nohup env DOUYIN_V82_SERVICE_ID="${id}_upstream" CUDA_VISIBLE_DEVICES="$service_gpu" \
      VLLM_PLUGINS=__none__ "${startup[@]}" >"$upstream_log" 2>&1 < /dev/null &
  fi
  echo "$!" > "$STATE_DIR/${id}_upstream.pid"
  wait_http "http://127.0.0.1:${up}/health" || {
    tail -n 80 "$upstream_log" >&2 || true
    stop_one "$id"
    return 1
  }
  startup_json="$(printf '%s\n' "${startup[@]}" | "$PYTHON_BIN" -c \
    'import json,sys; print(json.dumps([x.rstrip("\n") for x in sys.stdin]))')"
  proxy_args=("$PYTHON_BIN" "$DOUYIN_DIR/scripts/v82_flashvid_audit_proxy.py"
    --port "$pub" --upstream "http://127.0.0.1:$up" --retention-ratio "$keep"
    --backend "$service_backend" --service-git-dir "$FLASHVID_DIR"
    --core-file src/flashvid_vllm/audit.py
    --service-startup-args-json "$startup_json")
  if [[ "$id" == r025 ]]; then proxy_args+=(--model-audit-jsonl "$audit_file"); fi
  setsid nohup env DOUYIN_V82_SERVICE_ID="${id}_proxy" \
    "${proxy_args[@]}" >"$proxy_log" 2>&1 < /dev/null &
  echo "$!" > "$STATE_DIR/${id}_proxy.pid"
  wait_http "http://127.0.0.1:${pub}/health" || {
    tail -n 80 "$proxy_log" >&2 || true
    stop_one "$id"
    return 1
  }
  echo "started $id gpu=$service_gpu proxy=$pub upstream=$up"
}

command="${1:-status}"
case "$command" in
  start)
    [[ -f "$FLASHVID_DIR/src/flashvid_vllm/model.py" ]] || {
      echo "missing clean FlashVID V8.2 runtime: $FLASHVID_DIR" >&2; exit 2; }
    start_one r025
    start_one r100
    ;;
  stop) stop_one r025; stop_one r100 ;;
  status)
    for id in r025 r100; do
      state=stopped
      pidfile="$STATE_DIR/${id}_proxy.pid"
      if [[ -f "$pidfile" ]] && owned "${id}_proxy" "$(cat "$pidfile")"; then
        state="running proxy=$(cat "$pidfile") port=$(public_port "$id")"
      fi
      echo "$id $state"
    done
    ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
