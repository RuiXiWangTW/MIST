# MIST: Metabolite Inference with Spectrum Transformers

MIST predicts molecular fingerprints from tandem mass spectra using a chemical
formula transformer. This repository is an improved implementation of
[Samuel Goldman and colleagues' MIST](https://github.com/samgoldman97/mist),
with native PyTorch distributed training, compact memory-mapped data, and
optional training on synthetic spectra.

The model retains formula-aware attention and iterative fingerprint prediction.
It outputs 4,096-bit Morgan fingerprints (radius 2, without chirality). The
provided MSG and CANOPUS models have 72.3 million parameters. The training
implementation supports single-device execution, DDP, gradient accumulation,
BF16, EMA, and checkpoint recovery including per-rank random states and data
positions. CPU execution is available for development and small runs.

## Install

Use Python 3.10 or newer and PyTorch 2.7 or newer. Install the PyTorch build
appropriate for your hardware, then:

```bash
git clone https://github.com/coleygroup/MIST.git
cd MIST
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively, create the environment with `conda env create -f environment.yml`
and activate it with `conda activate mist` before installing the package.

GLACIER is an optional spectrum generator. Training on the prepared synthetic
caches requires only the core MIST installation. See
[GLACIER inference](docs/glacier.md) to generate spectra from molecules.

## Data and pretrained models

**Zenodo uploads are pending.** The download catalog contains explicit
placeholders for URLs, DOIs, and checksums. Model weights and datasets are not
included in Git. `mist-download` reports an unavailable artifact clearly until
its upload has been filled in.

| Dataset | Processed source data | Compact features | Synthetic features | MIST checkpoints | Generator |
|---|---|---|---|---|---|
| MassSpecGym (MSG) | `msg-source` | `msg-compact` | `msg-synthetic` (refined 4M molecules) | `msg-noaug`, `msg-aug` | `msg-glacier` |
| CANOPUS (NPLIB1) | `canopus-source` | `canopus-compact` | `canopus-synthetic` (random 2M molecules) | `canopus-noaug`, `canopus-aug` | `canopus-glacier` |

After the uploads become available:

```bash
mist-download --list
mist-download canopus-compact
mist-download canopus-noaug --output-dir checkpoints
tar -xzf downloads/canopus_compact.tar.gz
```

Compact data archives unpack into `data/compact/{msg,canopus}/{train,val,test}`.
Synthetic archives unpack into `data/compact/{msg,canopus}/synthetic`. They are
ready for training; rebuilding formula features is optional. The raw processed
archives contain the smaller HDF5-based representation of spectra and formula
assignments. [Data layout and preparation](docs/data.md) describes both forms.

## Train

There are four recipes:

| Recipe | Training data | Reference device count | Released filename |
|---|---|---:|---|
| `msg-noaug` | MSG real spectra | 1 | `mist_msg_noaug.ckpt` |
| `canopus-noaug` | CANOPUS real spectra | 1 | `mist_canopus_noaug.ckpt` |
| `msg-aug` | MSG real + refined-4M synthetic spectra | 4 | `mist_msg_aug.ckpt` |
| `canopus-aug` | CANOPUS real + random-2M synthetic spectra | 4 | `mist_canopus_aug.ckpt` |

```bash
scripts/train_recipe.sh canopus-noaug
scripts/train_recipe.sh msg-aug
```

Here **noaug means no synthetic training spectra**. Both recipe families retain
stochastic peak dropout and intensity perturbation as spectrum regularization.
Set `augmentation.spectrum_probability: 0.0` to disable that regularization too.
Validation always uses real held-out spectra. EMA checkpoints are selected by
validation fingerprint Tanimoto; the test set is not read during training.

The recipe launcher resumes `last.pt` if present and exports the
validation-selected `best.pt`. Files in `checkpoints/` contain the EMA model for
inference; full training checkpoints in `runs/` additionally contain optimizer,
scheduler, random states, and source positions.

To customize training, copy a YAML from `configs/noaug/` or `configs/aug/`, adjust
its paths and batch sizes, and set `train.expected_world_size` to your device
count (or remove that check). Batch sizes are **per rank, per microbatch**;
the global batch is their sum × world size × gradient accumulation. Changing
that product changes the training recipe.

```bash
NPROC_PER_NODE=2 scripts/train_ddp.sh my_config.yaml
mist-train --config my_config.yaml --device cpu
```

For multi-node execution, use your cluster's `torchrun` launch settings with
`-m mist.train_scaled --config my_config.yaml`. The convenience shell script
starts a single-node job. [Training details](docs/training.md) explains resume,
validation coverage, precision, and the published configurations.

## Predict fingerprints

```bash
mist-predict-fingerprints \
  --checkpoint checkpoints/mist_canopus_noaug.ckpt \
  --cache data/compact/canopus/test \
  --output runs/predictions/canopus \
  --allow-test-input
```

Use `--device cpu` for CPU inference. Outputs include `probabilities.npy` and
`spectrum_names.npy`, in matching row order, plus a provenance manifest.
Prediction reads spectrum features and names without opening fingerprint labels.

## Evaluate

Calibrate a bit threshold on validation, then apply it unchanged to test:

```bash
mist-evaluate \
  --checkpoint checkpoints/mist_canopus_noaug.ckpt \
  --cache data/compact/canopus/val --output runs/eval/canopus-val

mist-evaluate \
  --checkpoint checkpoints/mist_canopus_noaug.ckpt \
  --cache data/compact/canopus/test --output runs/eval/canopus-test \
  --threshold-file runs/eval/canopus-val/threshold.json --allow-test
```

The threshold file is bound to the checkpoint hash and dataset. Evaluation
reports spectrum-level and compound-level fingerprint Tanimoto metrics. Use
`--device cpu` for CPU evaluation or `scripts/evaluate_ddp.sh` for DDP.

## Development

```bash
python -m pip install -e '.[test]'
pytest -q
ruff check src scripts tests
```

## Citation

Please cite the original MIST paper:

> Goldman, S., Wohlwend, J., Stražar, M., Haroush, G., Xavier, R. J. & Coley, C. W.
> **Annotating metabolite mass spectra with domain-inspired chemical formula
> transformers.** Nature Machine Intelligence **5**, 965–979 (2023).
> [doi:10.1038/s42256-023-00708-3](https://doi.org/10.1038/s42256-023-00708-3)

```bibtex
@article{goldman2023mist,
  title = {Annotating metabolite mass spectra with domain-inspired chemical formula transformers},
  author = {Goldman, Samuel and Wohlwend, Jeremy and Stražar, Martin and
            Haroush, Guy and Xavier, Ramnik J. and Coley, Connor W.},
  journal = {Nature Machine Intelligence},
  volume = {5},
  pages = {965--979},
  year = {2023},
  doi = {10.1038/s42256-023-00708-3}
}
```

This implementation and its training recipes are engineering extensions of MIST;
the original paper establishes the method. See [NOTICE.md](NOTICE.md) for
attribution. Code is distributed under the [MIT license](LICENSE.md).
