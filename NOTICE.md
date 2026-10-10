# Attribution

MIST was introduced by Samuel Goldman, Jeremy Wohlwend, Martin Stražar,
Guy Haroush, Ramnik J. Xavier, and Connor W. Coley in *Annotating metabolite
mass spectra with domain-inspired chemical formula transformers*, Nature
Machine Intelligence 5, 965–979 (2023),
<https://doi.org/10.1038/s42256-023-00708-3>.

This repository builds on the MIT-licensed implementation at
<https://github.com/samgoldman97/mist>. Its copyright notice is preserved in
[LICENSE.md](LICENSE.md). The native PyTorch training loop, compact caches,
distributed execution, and synthetic-data recipes are later engineering
extensions and should not be attributed to the original paper.

The optional GLACIER inference code in `src/mist/glacier/` derives from
[ms-pred](https://github.com/coleygroup/ms-pred) and preserves the molecular
feature and intensity-decoder architecture used by the accompanying generator
checkpoints. The upstream copyright and MIT license are reproduced in
[LICENSES/ms-pred.txt](LICENSES/ms-pred.txt). GLACIER's method is described in
[GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection Problem](https://arxiv.org/abs/2606.29161).

Dataset and model downloads are separate artifacts; their Zenodo records will
provide artifact-specific provenance and license information.
