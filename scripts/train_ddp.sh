#!/usr/bin/env bash
set -euo pipefail

config=${1:-configs/noaug/msg.yaml}
if (( $# > 0 )); then
  shift
fi
nproc=${NPROC_PER_NODE:-1}
python_bin=${PYTHON_BIN:-python}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${nproc}" \
  -m mist.train_scaled \
  --config "${config}" \
  "$@"
