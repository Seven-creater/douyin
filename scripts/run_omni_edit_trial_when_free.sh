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

while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '[0-9]'; do
  echo "[$(date -Is)] waiting: GPU compute processes are still active"
  sleep 60
done

echo "[$(date -Is)] GPUs free; starting Omni editorial trial"
exec env PYTHONUNBUFFERED=1 "$python_bin" -m src.agentic_video.cli \
  omni-edit-trial \
  --spec config/experiments/lxh1_omni_edit_trial.json \
  --output "$output_dir" \
  --gpu-pairs "$gpu_pairs" \
  --worker-timeout 3600 \
  --force
