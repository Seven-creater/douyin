#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

repo_dir="/data02/usr/wangqihao/Demo/research"
python_bin="/data02/usr/wangqihao/miniconda3/envs/omni_src/bin/python"
output_dir="$1"
gpu_pairs="0,1;2,3;4,5;6,7"

cd "$repo_dir"
mkdir -p "$output_dir"
echo "[$(date -Is)] queued git=$(git rev-parse --short HEAD)"

gpu_memory_busy() {
  local snapshot value count
  snapshot="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    2>/dev/null || true)"
  count="$(printf '%s\n' "$snapshot" | sed '/^[[:space:]]*$/d' | wc -l)"
  [[ "$count" -eq 8 ]] || return 0
  while IFS= read -r value; do
    value="${value//[[:space:]]/}"
    [[ "$value" =~ ^[0-9]+$ ]] || return 0
    (( value >= 2000 )) && return 0
  done <<< "$snapshot"
  return 1
}

# Require three consecutive idle snapshots.  A single empty compute-process
# query has produced a false-free result on this shared server before.
idle_checks=0
while (( idle_checks < 3 )); do
  if gpu_memory_busy; then
    idle_checks=0
    echo "[$(date -Is)] waiting: one or more GPUs use at least 2 GiB"
  else
    ((idle_checks += 1))
    echo "[$(date -Is)] idle confirmation $idle_checks/3"
  fi
  (( idle_checks < 3 )) && sleep 30
done

echo "[$(date -Is)] GPUs free; starting Omni editorial trial"
exec env PYTHONUNBUFFERED=1 "$python_bin" -m src.agentic_video.cli \
  omni-edit-trial \
  --spec config/experiments/lxh1_omni_edit_trial.json \
  --output "$output_dir" \
  --gpu-pairs "$gpu_pairs" \
  --worker-timeout 3600 \
  --force
