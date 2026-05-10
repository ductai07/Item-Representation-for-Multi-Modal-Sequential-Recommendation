from __future__ import annotations

import torch

from cstamoerec.model import last_non_pad_indices


def test_last_non_pad_indices_left_padded_sequences() -> None:
    item_ids = torch.tensor(
        [
            [0, 0, 0, 4, 5, 6],
            [0, 0, 7, 8, 9, 10],
            [0, 0, 0, 0, 0, 11],
        ]
    )

    assert last_non_pad_indices(item_ids).tolist() == [5, 5, 5]


def test_last_non_pad_indices_internal_zero_is_not_selected() -> None:
    item_ids = torch.tensor([[0, 0, 4, 0, 5, 0]])

    assert last_non_pad_indices(item_ids).tolist() == [4]
