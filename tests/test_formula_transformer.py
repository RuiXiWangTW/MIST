import torch

from mist.models.modules import subset_formula_differences


def test_subset_formula_differences_zeroes_mixed_sign_pairs():
    differences = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [-1.0, -2.0, 0.0],
            [1.0, -2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ]
    )

    actual = subset_formula_differences(differences)

    expected = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_subset_formula_differences_does_not_mutate_input():
    differences = torch.tensor([[1.0, -2.0]])
    original = differences.clone()

    subset_formula_differences(differences)

    torch.testing.assert_close(differences, original, rtol=0.0, atol=0.0)
