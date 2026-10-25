#!/usr/bin/env bash
# Index Enveda subformula assignments, write compact features, then train.
#
# The real-cache step walks subformulae/no_subform.hdf5, assigns fragment
# formulas, and writes data/compact/enveda/{train,val,test}. An existing split
# manifest with the same source files is left in place.
#
# Training is configs/aug/enveda.yaml: real spectra plus the 2,000,000-molecule
# GLACIER cache at data/compact/enveda/synthetic. Set SIMULATION to that plan
# directory when the synthetic cache still has to be converted. The recipe
# expects four ranks; override with NPROC_PER_NODE.
#
#   DATA_DIR=$HOME/ms-pred-dev/data/spec_datasets scripts/enveda_aug.sh
#   SIMULATION=/path/to/simulation scripts/enveda_aug.sh
#   SKIP_TRAIN=1 scripts/enveda_aug.sh
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python_bin=${PYTHON_BIN:-python}

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -f "$HOME/ms-pred-dev/data/spec_datasets/enveda/labels.tsv" ]]; then
    DATA_DIR="$HOME/ms-pred-dev/data/spec_datasets"
  else
    DATA_DIR="data"
  fi
fi
data_dir=$DATA_DIR
enveda_root="$data_dir/enveda"
real_output="data/compact/enveda"
synthetic_cache="$real_output/synthetic"

if [[ ! -f "$enveda_root/labels.tsv" || ! -f "$enveda_root/subformulae/no_subform.hdf5" ]]; then
  echo "Enveda source data is missing under $data_dir (expected labels.tsv and subformulae/no_subform.hdf5)" >&2
  exit 2
fi

echo "indexing Enveda assignments and writing compact features to $real_output" >&2
"$python_bin" -m mist.data.build_all_cache \
  --dataset-name enveda \
  --data-dir "$data_dir" \
  --output "$real_output"

if [[ ! -f "$synthetic_cache/manifest.json" ]]; then
  if [[ -z "${SIMULATION:-}" ]]; then
    echo "real caches are ready; training also needs $synthetic_cache" >&2
    echo "convert the 2,000,000-molecule GLACIER plan with SIMULATION=/path/to/simulation $0" >&2
    exit 2
  fi
  echo "converting GLACIER predictions from $SIMULATION" >&2
  "$python_bin" -m mist.data.build_simulated_cache \
    --simulation "$SIMULATION" \
    --output "$synthetic_cache" \
    --workers "${SIM_WORKERS:-8}"
fi

if [[ "${SKIP_TRAIN:-0}" == 1 ]]; then
  echo "caches are ready; skipping training" >&2
  exit 0
fi

echo "training configs/aug/enveda.yaml" >&2
exec "$repo_root/scripts/train_recipe.sh" enveda-aug
