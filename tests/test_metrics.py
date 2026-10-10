import numpy as np
import torch

from mist.metrics import (
    calibrate_threshold,
    fingerprint_loss,
    fold_fingerprint,
    tanimoto_scores,
)


def test_modulo_fingerprint_fold():
    target = torch.zeros(1, 8)
    target[0, [0, 3, 4, 7]] = 1
    assert fold_fingerprint(target, 4).tolist() == [[1.0, 0.0, 0.0, 1.0]]


def test_threshold_calibration_and_tanimoto():
    probabilities = np.asarray([[0.9, 0.2, 0.1], [0.8, 0.7, 0.3]], dtype=np.float32)
    targets = np.asarray([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    threshold, stats = calibrate_threshold(probabilities, targets)
    assert stats["predicted_total_on_bits"] == stats["true_total_on_bits"] == 3
    assert tanimoto_scores(probabilities, targets, threshold).tolist() == [1.0, 1.0]


def test_combined_training_loss_backpropagates():
    prediction = torch.full((2, 8), 0.4, requires_grad=True)
    target = torch.zeros((2, 8))
    target[:, [1, 5]] = 1
    intermediate = [torch.full((2, 4), 0.3, requires_grad=True)]
    loss, parts = fingerprint_loss(
        prediction,
        target,
        intermediate,
        cosine_weight=1.0,
        tanimoto_weight=0.1,
        iterative_weight=0.4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert set(parts) == {
        "loss",
        "cosine_loss",
        "soft_tanimoto_loss",
        "iterative_loss",
    }
