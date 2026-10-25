#!/usr/bin/env bash
set -euo pipefail
recipe=${1:-}
case "$recipe" in
  msg-noaug|canopus-noaug|enveda-noaug|msg-aug|canopus-aug|enveda-aug) ;;
  *) echo "usage: $0 {msg-noaug|canopus-noaug|enveda-noaug|msg-aug|canopus-aug|enveda-aug}" >&2; exit 2 ;;
esac
dataset=${recipe%-*}
mode=${recipe##*-}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python_bin=${PYTHON_BIN:-python}
if [[ "$mode" == aug ]]; then default_nproc=4; else default_nproc=1; fi
export NPROC_PER_NODE=${NPROC_PER_NODE:-$default_nproc}
resume=()
if [[ -s "runs/$mode/$dataset/last.pt" ]]; then
  resume=(--resume "runs/$mode/$dataset/last.pt")
fi
"$repo_root/scripts/train_ddp.sh" "configs/$mode/$dataset.yaml" "${resume[@]}"
"$python_bin" "$repo_root/scripts/export_scaled_release.py" \
  --checkpoint "runs/$mode/$dataset/best.pt" \
  --output "checkpoints/mist_${dataset}_${mode}.ckpt"
