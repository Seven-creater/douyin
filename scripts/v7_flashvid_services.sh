#!/usr/bin/env bash
set -euo pipefail

# V7 owns only processes whose PID and environment marker are written below.
# The external FlashVID worktree is executed read-only; this script never pulls,
# resets, patches or stops services it did not start.

FLASHVID_DIR="${FLASHVID_DIR:-/data02/usr/wangqihao/Demo/test/flashvid}"
V7_STATE_DIR="${V7_STATE_DIR:-/data02/usr/wangqihao/Demo/research/data/agentic_runs/.v7_services}"
MEDIA_ROOT="${V7_MEDIA_ROOT:-/data02/usr/wangqihao/Demo}"
MODEL_PATH="${V7_MODEL_PATH:-${FLASHVID_DIR}/models/Qwen3.5-4B}"
V7_VLLM_BIN="${V7_VLLM_BIN:-${FLASHVID_DIR}/.venv/bin/vllm}"
V7_FLASHVID_BIN="${V7_FLASHVID_BIN:-${FLASHVID_DIR}/.venv/bin/flashvid-serve}"
V7_VLLM_PYTHON="${V7_VLLM_PYTHON:-}"
V7_FLASHVID_SITE_PACKAGES="${V7_FLASHVID_SITE_PACKAGES:-}"
V7_VANILLA_VLLM_PLUGINS="${V7_VANILLA_VLLM_PLUGINS:-__none__}"
mkdir -p "$V7_STATE_DIR"

service_port() {
  case "$1" in r010) echo 8101 ;; r025) echo 8102 ;; r100) echo 8104 ;; *) return 2 ;; esac
}

service_ratio() {
  case "$1" in r010) echo 0.10 ;; r025) echo 0.25 ;; r100) echo 1.00 ;; *) return 2 ;; esac
}

service_pruning_rate() {
  case "$1" in r010) echo 0.90 ;; r025) echo 0.75 ;; r100) echo 0.00 ;; *) return 2 ;; esac
}

service_gpu() {
  case "$1" in
    r010) echo "${V7_GPU_R010:-0}" ;;
    r025) echo "${V7_GPU_R025:-1}" ;;
    r100) echo "${V7_GPU_R100:-2}" ;;
    *) return 2 ;;
  esac
}

owned() {
  local id="$1" pid="$2"
  [[ -r "/proc/$pid/environ" ]] || return 1
  tr '\0' '\n' < "/proc/$pid/environ" | grep -Fxq "DOUYIN_V7_SERVICE_ID=$id"
}

start_one() {
  local id="$1" port ratio pruning_rate gpu pidfile logfile
  port="$(service_port "$id")"; ratio="$(service_ratio "$id")"; gpu="$(service_gpu "$id")"
  pruning_rate="$(service_pruning_rate "$id")"
  pidfile="$V7_STATE_DIR/$id.pid"; logfile="$V7_STATE_DIR/$id.log"
  if [[ -f "$pidfile" ]] && owned "$id" "$(cat "$pidfile")"; then
    echo "$id already running pid=$(cat "$pidfile")"
    return
  fi
  if [[ "$id" == r100 ]]; then
    if [[ -n "$V7_VLLM_PYTHON" ]]; then
      vllm_command=("$V7_VLLM_PYTHON" -m vllm.entrypoints.cli.main)
    else
      vllm_command=("$V7_VLLM_BIN")
    fi
    nohup setsid env VLLM_PLUGINS="$V7_VANILLA_VLLM_PLUGINS" DOUYIN_V7_SERVICE_ID="$id" \
      CUDA_VISIBLE_DEVICES="$gpu" "${vllm_command[@]}" serve "$MODEL_PATH" \
      --served-model-name Qwen3.5-4B --default-chat-template-kwargs '{"enable_thinking":false}' \
      --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --dtype bfloat16 \
      --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.90 --enable-prompt-tokens-details \
      --limit-mm-per-prompt '{"image":4,"video":1}' \
      --allowed-local-media-path "$MEDIA_ROOT" >"$logfile" 2>&1 < /dev/null &
  else
    if [[ -n "$V7_VLLM_PYTHON" && -n "$V7_FLASHVID_SITE_PACKAGES" ]]; then
      nohup setsid env DOUYIN_V7_SERVICE_ID="$id" CUDA_VISIBLE_DEVICES="$gpu" \
        VLLM_PLUGINS=flashvid_qwen3_5 FLASHVID_VISION_RETENTION_RATIO="$ratio" \
        PYTHONPATH="$V7_FLASHVID_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" \
        "$V7_VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
        --hf-overrides '{"architectures":["FlashVIDQwen3_5ForConditionalGeneration"]}' \
        --video-pruning-rate "$pruning_rate" --served-model-name Qwen3.5-4B \
        --default-chat-template-kwargs '{"enable_thinking":false}' --host 127.0.0.1 \
        --port "$port" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 32768 \
        --max-num-seqs 8 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.90 \
        --enable-prompt-tokens-details --limit-mm-per-prompt '{"image":4,"video":1}' \
        --allowed-local-media-path "$MEDIA_ROOT" >"$logfile" 2>&1 < /dev/null &
    else
      nohup setsid env DOUYIN_V7_SERVICE_ID="$id" CUDA_VISIBLE_DEVICES="$gpu" \
        VLLM_PLUGINS=flashvid_qwen3_5 "$V7_FLASHVID_BIN" "$MODEL_PATH" \
        --vision-retention-ratio "$ratio" --served-model-name Qwen3.5-4B \
        --default-chat-template-kwargs '{"enable_thinking":false}' --host 127.0.0.1 \
        --port "$port" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 32768 \
        --max-num-seqs 8 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.90 \
        --enable-prompt-tokens-details --limit-mm-per-prompt '{"image":4,"video":1}' \
        --allowed-local-media-path "$MEDIA_ROOT" >"$logfile" 2>&1 < /dev/null &
    fi
  fi
  echo "$!" > "$pidfile"
  echo "started $id pid=$! gpu=$gpu port=$port ratio=$ratio"
}

stop_one() {
  local id="$1" pidfile pid
  pidfile="$V7_STATE_DIR/$id.pid"
  [[ -f "$pidfile" ]] || return 0
  pid="$(tr -d '[:space:]' < "$pidfile")"
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && owned "$id" "$pid"; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
  rm -f -- "$pidfile"
}

command="${1:-status}"
shift || true
ids=("${@:-r010 r025 r100}")
case "$command" in
  start) for id in ${ids[*]}; do start_one "$id"; done ;;
  stop) for id in ${ids[*]}; do stop_one "$id"; done ;;
  status)
    for id in r010 r025 r100; do
      pidfile="$V7_STATE_DIR/$id.pid"
      if [[ -f "$pidfile" ]] && owned "$id" "$(cat "$pidfile")"; then
        echo "$id running pid=$(cat "$pidfile") port=$(service_port "$id")"
      else echo "$id stopped"; fi
    done ;;
  *) echo "usage: $0 start|stop|status [r010 r025 r100]" >&2; exit 2 ;;
esac
