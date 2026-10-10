from argparse import Namespace
import json

import numpy as np
import torch

from mist.data.input_signal import predict_input_signal
from mist.evaluate_scaled import evaluate
from mist.models.scaled_mist import ScaledMIST
from test_training_sources import _write_cache


def test_validation_calibration_and_spectrum_only_prediction(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = {"model": {"hidden_size": 16, "num_heads": 2, "peak_attn_layers": 1, "refine_layers": 1},
                  "data": {"dataset_name": "canopus_all"}}
        model = ScaledMIST(**config["model"]).eval()
        path = tmp_path / "model.ckpt"
        torch.save({"config": config, "step": 1, "ema_model": model.state_dict()}, path)
        val = _write_cache(tmp_path / "val", source_type="real", split="val", rows=3)
        test = _write_cache(tmp_path / "test", source_type="real", split="test", rows=4)
        def args(cache, output, threshold_file=None):
            return Namespace(checkpoint=path, cache=cache, output=output, threshold_file=threshold_file,
                             allow_test=threshold_file is not None, raw_model=False, batch_size=2,
                             num_workers=0, prefetch_factor=2, device="cpu")
        evaluate(args(val, tmp_path / "eval_val"))
        threshold = tmp_path / "eval_val/threshold.json"
        evaluate(args(test, tmp_path / "eval_test", threshold))
        summary = json.loads((tmp_path / "eval_test/summary.json").read_text())
        assert summary["threshold"] == json.loads(threshold.read_text())["threshold"]
        # Prediction must succeed with the target arrays entirely absent.
        (test / "part_00000/fingerprints.npy").unlink()
        (test / "part_00000/structure_ids.npy").unlink()
        output = tmp_path / "predictions"
        predict_input_signal(checkpoint_path=path, cache_path=test, output=output,
                             allow_test_input=True, batch_size=2, num_workers=0, device_name="cpu")
        assert np.load(output / "probabilities.npy").shape == (4, 4096)
        assert np.load(output / "spectrum_names.npy").tolist() == ['s0', 's1', 's2', 's3']
    finally:
        torch.set_num_threads(previous)
