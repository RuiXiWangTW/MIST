import torch

from mist.data.augmentation import augment_peak_batch


def test_peak_augmentation_compacts_and_preserves_cls():
    torch.manual_seed(3)
    batch = {
        "form_vec": torch.arange(2 * 5 * 18).reshape(2, 5, 18).float(),
        "ion_vec": torch.zeros(2, 5, dtype=torch.long),
        "types": torch.tensor([[0, 0, 0, 3, 0], [0, 0, 3, 0, 0]]),
        "intens": torch.tensor([[1.0, 0.8, 0.5, 1.0, 0.0], [1.0, 0.2, 1.0, 0.0, 0.0]]),
        "num_peaks": torch.tensor([4, 3]),
        "instruments": torch.tensor([2, 3]),
        "targets": torch.zeros(2, 4096),
        "row_indices": torch.tensor([4, 8]),
    }
    output = augment_peak_batch(
        batch,
        spectrum_probability=1.0,
        peak_drop_probability=0.8,
        intensity_jitter_std=0.2,
    )
    positions = torch.arange(5)[None, :]
    valid = positions < output["num_peaks"][:, None]
    assert torch.all(((output["types"] == 3) & valid).sum(dim=1) == 1)
    assert torch.all(output["intens"][output["types"] == 3] == 1)
    assert torch.equal(output["targets"], batch["targets"])
    assert torch.equal(output["row_indices"], batch["row_indices"])
