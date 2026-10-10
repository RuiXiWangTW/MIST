import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml
from test_training_sources import _config

from mist.train_scaled import train


def tiny_config(tmp_path, synthetic):
    return {
        "seed": 73,
        "data": _config(tmp_path, synthetic),
        "model": {
            "hidden_size": 16,
            "num_heads": 2,
            "peak_attn_layers": 1,
            "refine_layers": 1,
            "spectra_dropout": 0.1,
            "output_size": 4096,
        },
        "loss": {"cosine_weight": 1.0, "tanimoto_weight": 0.1, "iterative_weight": 0.4},
        "augmentation": {
            "spectrum_probability": 0.5,
            "peak_drop_probability": 0.1,
            "intensity_jitter_std": 0.2,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_steps": 1,
        },
        "train": {
            "max_steps": 4,
            "lr_schedule_total_steps": 4,
            "gradient_accumulation": 2,
            "gradient_clip_norm": 5.0,
            "precision": "fp32",
            "ema_decay": 0.9,
            "validate_every": 2,
            "checkpoint_every": 2,
            "log_every": 2,
            "output_dir": str(tmp_path / "unused"),
        },
    }


@pytest.mark.parametrize("synthetic", [False, True])
def test_exact_resume_matches_uninterrupted_training(tmp_path, synthetic):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = tiny_config(tmp_path, synthetic)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))

        def run(output, resume=None, stop_after=None):
            train(
                Namespace(
                    config=path,
                    output_dir=output,
                    resume=resume,
                    device="cpu",
                    stop_after=stop_after,
                )
            )

        run(tmp_path / "full")
        run(tmp_path / "resumed", stop_after=2)
        run(tmp_path / "resumed", resume=tmp_path / "resumed/last.pt")
        full = torch.load(tmp_path / "full/last.pt", weights_only=False)
        resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
        for state in ["model", "ema_model"]:
            for key in full[state]:
                torch.testing.assert_close(
                    full[state][key], resumed[state][key], rtol=0, atol=0
                )
        assert (
            full["runtime_states"][0]["sources"]
            == resumed["runtime_states"][0]["sources"]
        )
    finally:
        torch.set_num_threads(previous)


def test_two_rank_training_with_empty_validation_rank(tmp_path):
    config = tiny_config(tmp_path, True)
    config["train"].update(max_steps=2, output_dir=str(tmp_path / "ddp"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "mist.train_scaled",
            "--config",
            str(path),
            "--device",
            "cpu",
        ],
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout
    checkpoint = torch.load(tmp_path / "ddp/last.pt", weights_only=False)
    assert len(checkpoint["runtime_states"]) == 2
    run = json.loads((tmp_path / "ddp/run_manifest.json").read_text())
    assert run["world_size"] == 2
