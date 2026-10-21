# Training and checkpoint recovery

`mist-train` reads the real and optional synthetic sources declared in a YAML
configuration. Each source contributes its configured per-rank microbatch on
every step. Real-only training uses one source; augmented training concatenates
the real and simulated microbatches before a shared forward pass.

## Reference recipes

| Dataset / mode | Real batch per rank | Synthetic batch per rank | Ranks | Accumulation | Maximum steps |
|---|---:|---:|---:|---:|---:|
| MSG / noaug | 77 | 0 | 1 | 4 | 30,000 |
| CANOPUS / noaug | 115 | 0 | 1 | 4 | 30,000 |
| MSG / aug | 75 | 2,048 | 4 | 1 | 478,850 |
| CANOPUS / aug | 225 | 2,048 | 4 | 1 | 244,150 |

The noaug configurations stop at 30,000 steps within a 120,000-step cosine
schedule; their stopping point is distinct from the schedule horizon.
The augmented configurations use a 50-synthetic-spectrum-epoch budget.
The release launcher exports the checkpoint with the best validation score,
which can occur before the final step.

Configuration files specify the seed, optimizer, warmup, EMA decay, precision,
and validation cadence. They describe the training recipes. The released EMA
checkpoints are available on [Zenodo](https://zenodo.org/records/23303289);
each checkpoint includes its model configuration, dataset, and selected step.
Editing the world size or source batch sizes changes data exposure. For a customized run, record the
modified configuration and recalibrate on its validation set.

## Distributed data loading

The locality sampler reads contiguous blocks from memory-mapped parts. It
shuffles part/block order deterministically and rotates rank ownership across
epochs. Training ranks receive disjoint, equally numerous batches, dropping
incomplete distributed tails. `combine_part_tails: true` allows small synthetic
part tails to form complete batches. Each source cycles independently.

Validation and evaluation retain every row exactly once, without padding with
duplicate examples. A rank with no validation rows participates in the gather
with an empty tensor. Threshold calibration occurs on rank 0 after gathering
all validation predictions.

DDP synchronizes gradients; EMA starts from the synchronized model and is
updated after every optimizer step. Gradient accumulation uses `no_sync()` on
intermediate microbatches. Set `ddp_static_graph: false` when accumulating
across multiple DDP microbatches. This is selected automatically unless the
configuration overrides it.

## Precision and hardware

Training supports `bf16` and `fp32`. Use BF16-capable GPUs for the reference
recipes. Evaluation and threshold calibration use FP32. CUDA uses NCCL;
CPU development runs use Gloo and disable CUDA-only fused optimizer execution.
`--device auto` selects CUDA when available. Multi-node launching is delegated
to `torchrun`; the model uses `LOCAL_RANK` for the local device and records each
rank's hardware in its run manifest.

### Blackwell and older NCCL builds

The two-node RTX PRO 6000 smoke test passed with PyTorch 2.7.0/CUDA 12.8
and NCCL 2.26.2 using `NCCL_PROTO=LL`. With this older NCCL build, the default
protocol failed during large parameter synchronization; small collectives
passed. This matches the [reported NCCL/Blackwell issue](https://github.com/NVIDIA/nccl/issues/2418).
Use a CUDA/NCCL build compatible with the GPU. For the affected environment,
set `NCCL_PROTO=LL` in every rank's environment as a temporary workaround;
it can reduce communication throughput. MIST does not change NCCL settings
inside the training process.

## Resume

```bash
mist-train --config my_config.yaml --resume runs/my-run/last.pt
```

A full checkpoint contains raw and EMA models, optimizer and scheduler states,
best validation score, and per-rank Python/NumPy/PyTorch random states and
source cursors. Resume requires the same configuration, data manifests,
world size, and hardware identity. DataLoader prefetch does not move the saved
logical cursor: it tracks batches consumed by the training loop. Deterministic
resume tests compare both raw and EMA weights against uninterrupted execution.
GPU floating-point kernels can still have hardware-dependent nondeterminism.

To test an interruption without changing the learning-rate schedule:

```bash
mist-train --config my_config.yaml --stop-after 100
mist-train --config my_config.yaml --resume runs/my-run/last.pt
```

EMA release files and older training files lacking per-rank runtime states
cannot provide exact training recovery. Use `last.pt` written by this trainer.
