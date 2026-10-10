"""Vectorized peak dropout and intensity jitter for fixed-width MIST batches."""

from __future__ import annotations

import torch


def augment_peak_batch(
    batch: dict[str, torch.Tensor],
    *,
    spectrum_probability: float,
    peak_drop_probability: float,
    intensity_jitter_std: float,
) -> dict[str, torch.Tensor]:
    """Augment spectra on-device while always retaining and compacting CLS."""

    if not 0.0 <= spectrum_probability <= 1.0:
        raise ValueError("spectrum_probability must be in [0, 1]")
    if not 0.0 <= peak_drop_probability < 1.0:
        raise ValueError("peak_drop_probability must be in [0, 1)")
    if intensity_jitter_std < 0.0:
        raise ValueError("intensity_jitter_std must be nonnegative")
    if spectrum_probability == 0.0 or (
        peak_drop_probability == 0.0 and intensity_jitter_std == 0.0
    ):
        return batch

    batch_size, width = batch["types"].shape
    positions = torch.arange(width, device=batch["types"].device)[None, :]
    valid = positions < batch["num_peaks"][:, None]
    fragments = valid & (batch["types"] == 0)
    cls = valid & (batch["types"] == 3)
    if not torch.all(cls.sum(dim=1) == 1):
        raise ValueError("every spectrum must contain exactly one valid CLS row")
    selected = torch.rand(batch_size, device=positions.device) < spectrum_probability

    intensities = batch["intens"].clone()
    if intensity_jitter_std:
        factors = torch.exp(torch.randn_like(intensities) * intensity_jitter_std)
        jitter_mask = fragments & selected[:, None]
        intensities = torch.where(jitter_mask, intensities * factors, intensities)
        fragment_max = torch.where(fragments, intensities, 0.0).amax(dim=1, keepdim=True)
        normalized = intensities / fragment_max.clamp_min(1e-8)
        normalize_mask = fragments & selected[:, None] & (fragment_max > 0)
        intensities = torch.where(normalize_mask, normalized, intensities)
        intensities = torch.where(cls, torch.ones_like(intensities), intensities)

    drop = torch.zeros_like(valid)
    if peak_drop_probability:
        drop = (
            fragments
            & selected[:, None]
            & (torch.rand(batch_size, width, device=positions.device) < peak_drop_probability)
        )
    keep = valid & ~drop
    # Stable integer keys compact retained rows while preserving peak order.
    sort_key = (~keep).to(torch.int64) * width + positions
    order = torch.argsort(sort_key, dim=1)

    output = dict(batch)
    output["num_peaks"] = keep.sum(dim=1).to(batch["num_peaks"].dtype)
    output["form_vec"] = torch.gather(
        batch["form_vec"], 1, order[:, :, None].expand(-1, -1, batch["form_vec"].shape[-1])
    )
    output["ion_vec"] = torch.gather(batch["ion_vec"], 1, order)
    output["types"] = torch.gather(batch["types"], 1, order)
    output["intens"] = torch.gather(intensities, 1, order)
    return output
