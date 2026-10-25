#!/usr/bin/env python3
"""Export a portable EMA inference checkpoint without training paths or optimizer state."""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch

from mist.data.simulation_utils import sha256_file
from mist.data.datasets import SUPPORTED_DATASETS
from mist.models.scaled_mist import ScaledMIST, count_parameters, load_mist_weights


def export_checkpoint(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False, mmap=True)
        if existing.get("source_checkpoint_sha256") == sha256_file(source):
            return
        raise FileExistsError(f"existing release was exported from a different checkpoint: {output}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if not {"ema_model", "step", "config"} <= checkpoint.keys():
        raise ValueError("checkpoint must contain ema_model, step and config")
    original = checkpoint["config"]
    model_config = deepcopy(original["model"])
    model = ScaledMIST(**model_config)
    load_mist_weights(model, checkpoint["ema_model"])
    dataset = original["data"]["dataset_name"]
    if dataset not in SUPPORTED_DATASETS:
        choices = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"checkpoint dataset must be one of: {choices}")
    synthetic = original.get("experiment", {}).get("synthetic_training", any(s["source_type"] == "simulated" for s in original["data"].get("training_sources", [])))
    # Retain only configuration needed for inference and scientific interpretation.
    config = {
        "model": model_config,
        "data": {"dataset_name": dataset},
        "experiment": {"release_filename": output.name, "parameter_count": count_parameters(model),
                       "synthetic_training": synthetic},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    try:
        torch.save({
            "checkpoint_kind": "mist_ema_release", "step": int(checkpoint["step"]),
            "ema_model": checkpoint["ema_model"], "config": config,
            "source_checkpoint_sha256": sha256_file(source),
        }, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
