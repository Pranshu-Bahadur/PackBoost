"""Dataset helpers for PackBoost FAST PATH era sharding."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def group_rows_by_era(era_ids: np.ndarray, n_eras: int) -> Tuple[np.ndarray, ...]:
    """Return row indices grouped by era in stable order."""
    if era_ids.ndim != 1:
        raise ValueError("era_ids must be a 1D array")
    if n_eras <= 0:
        raise ValueError("n_eras must be positive")

    era_ids_int = np.asarray(era_ids, dtype=np.int32)
    if era_ids_int.size == 0:
        empty = np.empty(0, dtype=np.int32)
        return tuple(empty for _ in range(n_eras))

    counts = np.bincount(era_ids_int, minlength=n_eras)
    if counts.shape[0] < n_eras:
        counts = np.pad(counts, (0, n_eras - counts.shape[0]))

    order = np.argsort(era_ids_int, kind="stable")
    order = order.astype(np.int32, copy=False)

    offsets = np.empty(n_eras + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts[:n_eras], out=offsets[1:])

    grouped: list[np.ndarray] = []
    for era in range(n_eras):
        start = int(offsets[era])
        end = int(offsets[era + 1])
        grouped.append(order[start:end])
    return tuple(grouped)
