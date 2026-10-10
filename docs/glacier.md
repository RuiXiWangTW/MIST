# Optional GLACIER inference

GLACIER predicts spectra from molecules. MIST predicts molecular fingerprints
from spectra. GLACIER is needed only to generate new spectra; training MIST on
prepared synthetic caches does not import or require `ms_pred`.

## Public dependency and checkpoint compatibility

Use the public [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred)
repository. The tested revision is
`708148c2a8eb0c32120644436fefd2fe90cd2214`.

The accompanying MSG and CANOPUS generator weights use membership-aware
intensity attention and crossing-bond features. The public revision's standard
`JointModel` does not contain those layers: directly loading these weights into
it fails strict state loading (including a token-mapper width of 855 versus
551). Changing the checkout directory name cannot fix an architecture mismatch.

`mist.glacier` supplies the necessary inference architecture and molecular
features while using the public `ms_pred` Graphormer, chemistry, and tensor
utilities. `load_glacier` selects the matching architecture and always uses
strict state loading. It does not discard incompatible tensors or silently
initialize missing layers. The compatibility implementation supports the
Graphormer generator checkpoints; it is an inference component.

## Environment

Install the public `ms-pred` dependencies in a separate generator environment,
following its installation instructions for your hardware, and pin the checkout:

```bash
git clone https://github.com/coleygroup/ms-pred.git
cd ms-pred
git checkout 708148c2a8eb0c32120644436fefd2fe90cd2214
python -m pip install -e .
```

Install MIST into that environment as well. The generator additionally needs the
DGL and PyTorch Geometric dependencies described by `ms-pred`. Checkpoint and
prediction checks passed with Python 3.10, PyTorch 2.7.0,
RDKit 2025.3.6 (the upstream dependency series) and 2026.3.1, DGL 1.1.3,
torch-scatter 2.1.2, NumPy 2.2.6, and PyTorch Lightning 1.9.5. Select matching
binary wheels for your platform. MIST's core installation does not install
these optional graph libraries.

## Predict

After downloading the generator weights into `checkpoints/`:

```bash
mist-glacier-predict \
  --checkpoint checkpoints/glacier_canopus.ckpt \
  --smiles 'CC(=O)Oc1ccccc1C(=O)O' \
  --energy 20 --adduct '[M+H]+' --instrument Orbitrap \
  --device cpu --output runs/glacier/aspirin.npz
```

Use `--device cuda:0` for GPU inference. Collision energy is in eV, not normalized
collision energy. Supported release settings are `[M+H]+` or `[M+Na]+` and
Orbitrap or QTOF. SMILES are canonicalized before graph construction. The NPZ
contains `spec` (mass/intensity rows) and aligned `frag` atom-membership masks.
Multiple predicted fragments can yield the same mass; these are unbinned
fragment-level predictions, rather than a merged experimental spectrum.

Programmatic use:

```python
from pathlib import Path
from mist.glacier.predict import load_glacier, predict_spectrum

model = load_glacier(Path('checkpoints/glacier_canopus.ckpt'), device='cpu')
result = predict_spectrum(model, 'CCO', 20., '[M+H]+', 'Orbitrap', 'cpu')
```

The compatibility check includes strict loading of real generator weights and
small-molecule predictions against the generation implementation. It does not
claim that every future `ms-pred` revision is checkpoint compatible. Preserve
the pinned revision when reproducing synthetic spectra.
