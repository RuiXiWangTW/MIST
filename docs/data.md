# Data layout and preparation

The prepared feature caches are the shortest path to training. Source spectra
are also provided as compact HDF5 files for rebuilding features. Download
locations are listed in `src/mist/artifacts.json`; all URLs, SHA-256 values, and
Zenodo DOIs remain `null` until the corresponding artifact is uploaded.

## Archive layout

Extract archives from the repository root. Each archive should contain these
relative paths:

| Artifact | Contents |
|---|---|
| `msg-source`, `canopus-source` | `data/{msg_all,canopus_all}/` |
| `msg-compact`, `canopus-compact` | `data/compact/{msg,canopus}/{train,val,test}/` |
| `msg-synthetic`, `canopus-synthetic` | `data/compact/{msg,canopus}/synthetic/` |
| MIST checkpoints | Individual `mist_{msg,canopus}_{noaug,aug}.ckpt` files |
| GLACIER checkpoints | Individual `glacier_{msg,canopus}.ckpt` files |

Download checkpoint files directly into `checkpoints/`. The downloader verifies
the published SHA-256 before making a file available. It does not extract
archives. The source-data and compact-feature downloads are alternatives for
preparing real training inputs; neither recipe needs both copies.

## Real spectrum sources

```text
data/msg_all/
  labels.tsv
  spec_files.hdf5
  subformulae/no_subform.hdf5
  splits/split.tsv

data/canopus_all/
  labels.tsv
  spec_files.hdf5
  subformulae/no_subform.hdf5
  splits/source_splits/canopus_hplus_100_0.tsv
```

The label table contains `spec`, `formula`, `ionization`, `instrument`, `smiles`,
and `inchikey`. Split tables have exactly two columns, `name` and `split`, with
`train`, `val`, `test`, or `exclude` assignments. Formula-assignment HDF5 keys
identify spectrum/collision entries. Missing collision-energy labels do not
prevent feature construction. A precursor formula is required.

```bash
mist-build-real-cache --dataset-name msg_all \
  --data-dir data --output data/compact/msg
mist-build-real-cache --dataset-name canopus_all \
  --data-dir data --output data/compact/canopus
```

By default the builder retains all assigned collision entries and the ten
highest-intensity assigned fragments, plus a precursor/CLS token. The release
recipes use fixed cache selections. In particular, CANOPUS validation uses
one retained collision row per source spectrum (401 rows). A newly built cache
containing every collision row is a different validation set; do not substitute
it when reproducing a released model's validation selection. Use the published
compact archive to retain the release selection exactly.

## Compact feature contract

Every cache contains `manifest.json` and one or more part directories. Paths
in a downloadable cache are relative to that cache root. Manifests declare the
dataset, split, row count, `max_peaks`, fingerprint definition, and source type.
A synthetic training manifest must have `conditioning: {kind: none}`.

For `N` rows and `W = max_peaks + 1` tokens:

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `form_vec.npy` | `[N, W, 18]` | `int16` | Element counts |
| `ion_vec.npy` | `[N, W]` | `uint8` | Adduct category |
| `intens.npy` | `[N, W]` | `float16` | Normalized intensity |
| `peak_type.npy` | `[N, W]` | `uint8` | Fragment (`0`) or CLS (`3`) |
| `lengths.npy` | `[N]` | `uint8` | Valid token count |
| `instruments.npy` | `[N]` | `uint8` | Instrument category |
| `fingerprints.npy` | `[N, 512]` | `uint8` | Packed 4,096-bit Morgan targets |
| `names.npy` | `[N]` | Unicode | Spectrum identifiers (real caches) |
| `structure_ids.npy` | `[N]` | Unicode | Connectivity identifiers (real caches) |

The element order is `C H As B Br Cl Co F Fe I K N Na O P S Se Si`.
Fingerprint bytes use NumPy's little-endian bit order. Each spectrum has exactly
one valid CLS token. Training sources and validation must share the feature
layout. Spectrum-only inference does not read `fingerprints.npy` or
`structure_ids.npy`.

## Synthetic training spectra

The MSG recipe combines real spectra with the refined-4M molecule collection.
Its prepared cache retains 78,450,500 spectra after removing 40 rows with
unrepresentable formula counts. The CANOPUS recipe uses two million sampled
molecules, excluding validation and test connectivity identities before
sampling. Its 20 settings per molecule combine two adducts (`[M+H]+`, `[M+Na]+`),
two instruments (Orbitrap, QTOF), and five collision energies (10–50 eV), yielding
40 million spectra. These counts describe spectra and molecules separately.

Training reads the prepared arrays directly. To convert an existing fixed
GLACIER simulation plan with molecule catalogs, masses, intensities, and
fragment masks into the same representation:

```bash
mist-build-simulated-cache --simulation /path/to/simulation \
  --output data/compact/canopus/synthetic --workers 8
```

The plan must be `glacier_fixed_pretraining_plan`, schema 1, with a dataset
name and `conditioning: {kind: none}`. It specifies `work_units`, catalog
parts, settings, and the expected prediction count. Predictions are read from
`predictions/work_NNNNNN/`. The converter reconstructs fragment formulas from
atom masks and masses, checks their mass agreement, and fingerprints the
candidate molecules. For parallel conversion, give each process a distinct
`--rank` in `0..world-size-1`; after all ranks finish, publish the root manifest
using the same command with `--finalize-only`. The single-process default
finalizes automatically. This multiprocessing converter requires Linux.

## Preparing a Zenodo upload

Local working caches may use indexed views or external part directories.
Materialize them before packaging:

```bash
mist-export-cache --source /path/to/working/cache \
  --output staging/data/compact/canopus/val
```

This preserves logical rows and their order, copies only the required arrays
and spectrum metadata, materializes indices, and writes a self-contained
manifest. It drops local paths and records the input manifest's hash. Export
each split separately and archive `data/compact/canopus/` from `staging/`.
For a MIST weight file, use `scripts/export_scaled_release.py` to retain the EMA,
model configuration, dataset, step, and source hash while removing local paths,
optimizer state, and other training metadata.

After uploading, replace the matching `null` fields in
`src/mist/artifacts.json` with the actual HTTPS URL, DOI, and lowercase SHA-256.
Keep these fields unset for any artifact that is still unavailable. Include
source data provenance, split definitions, and artifact-specific licenses in
the Zenodo record.
