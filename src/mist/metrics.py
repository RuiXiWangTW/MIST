"""Losses and locked spec2fp metrics shared by training and evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def fold_fingerprint(target: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Fold bits by modulo, matching historical MIST iterative targets."""

    final_dim = target.shape[-1]
    if final_dim % output_dim:
        raise ValueError(f"cannot fold {final_dim} bits into {output_dim}")
    factor = final_dim // output_dim
    return target.reshape(target.shape[0], factor, output_dim).amax(dim=1)


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)


def soft_tanimoto_distance(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=-1)
    union = (prediction + target - prediction * target).sum(dim=-1)
    return 1.0 - intersection / union.clamp_min(1e-8)


def fingerprint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    intermediate_predictions: list[torch.Tensor],
    *,
    cosine_weight: float,
    tanimoto_weight: float,
    iterative_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cosine = cosine_distance(prediction, target).mean()
    tanimoto = soft_tanimoto_distance(prediction, target).mean()
    final_loss = cosine_weight * cosine + tanimoto_weight * tanimoto

    if intermediate_predictions:
        iterative_terms = []
        for intermediate in intermediate_predictions:
            folded = fold_fingerprint(target, intermediate.shape[-1])
            iterative_terms.append(cosine_distance(intermediate, folded).mean())
        # Historical MIST adds the loss from every folded unfolding stage
        # before applying iterative_weight; preserve that scale exactly.
        iterative = torch.stack(iterative_terms).sum()
    else:
        iterative = prediction.new_zeros(())
    total = final_loss + iterative_weight * iterative
    return total, {
        "loss": total.detach(),
        "cosine_loss": cosine.detach(),
        "soft_tanimoto_loss": tanimoto.detach(),
        "iterative_loss": iterative.detach(),
    }


def calibrate_threshold(
    probabilities: np.ndarray, targets: np.ndarray
) -> tuple[float, dict[str, float | int | bool]]:
    """Match total validation on-bit frequency with one scalar threshold."""

    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and targets must be aligned 2-D arrays")
    flat = probabilities.reshape(-1)
    desired_on = int(np.asarray(targets, dtype=np.uint64).sum())
    if not 0 < desired_on < flat.size:
        raise ValueError(f"invalid target on-bit count: {desired_on}")
    boundary = flat.size - desired_on
    partitioned = np.partition(flat, (boundary - 1, boundary))
    largest_off = float(partitioned[boundary - 1])
    smallest_on = float(partitioned[boundary])
    threshold = (
        (largest_off + smallest_on) / 2.0
        if smallest_on > largest_off
        else smallest_on
    )
    predicted_on = int(np.count_nonzero(probabilities >= threshold))
    exact = predicted_on == desired_on
    return threshold, {
        "threshold": threshold,
        "n_spectra": int(probabilities.shape[0]),
        "fingerprint_dim": int(probabilities.shape[1]),
        "true_total_on_bits": desired_on,
        "predicted_total_on_bits": predicted_on,
        "true_on_bit_frequency": desired_on / int(flat.size),
        "predicted_on_bit_frequency": predicted_on / int(flat.size),
        "exact_total_on_bit_match": exact,
        "largest_excluded_probability": largest_off,
        "smallest_included_probability": smallest_on,
    }


def tanimoto_scores(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> np.ndarray:
    predictions = probabilities >= threshold
    truths = targets.astype(bool, copy=False)
    intersection = np.logical_and(predictions, truths).sum(axis=1)
    union = np.logical_or(predictions, truths).sum(axis=1)
    return (intersection / np.maximum(union, 1)).astype(np.float64)


def metric_summary(scores: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a nonempty vector")
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values))
    mean = float(values.mean())
    return {
        "n_spectra": int(len(values)),
        "mean_tanimoto": mean,
        "median_tanimoto": float(np.median(values)),
        "std_tanimoto": std,
        "sem_tanimoto": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "min_tanimoto": float(values.min()),
        "max_tanimoto": float(values.max()),
    }
