"""Era shard data structures for the PackBoost FAST PATH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

_EMPTY = np.empty(0, dtype=np.int32)


@dataclass(frozen=True)
class EraShard:
    """Collection of row indices grouped by era."""

    rows_per_era: Tuple[np.ndarray, ...]

    @classmethod
    def from_grouped_indices(cls, grouped: Sequence[np.ndarray]) -> "EraShard":
        rows = tuple(np.asarray(arr, dtype=np.int32) for arr in grouped)
        return cls(rows)

    @classmethod
    def empty(cls, n_eras: int) -> "EraShard":
        return cls(tuple(_EMPTY for _ in range(n_eras)))

    @property
    def n_eras(self) -> int:
        return len(self.rows_per_era)

    @property
    def sample_count(self) -> int:
        return int(sum(int(rows.size) for rows in self.rows_per_era))


@dataclass
class NodeState:
    """State carried for each frontier node during training."""

    node_id: int
    shard: EraShard
    grad_sum: float
    hess_sum: float
    sample_count: int
    depth: int


@dataclass
class FrontierDecision:
    """Best split outcome for a frontier node."""

    feature: int | None
    threshold: int
    score: float
    agreement: float
    left_value: float
    right_value: float
    base_value: float
    left_grad: float
    left_hess: float
    left_count: int


def split_shard(shard: EraShard, X_binned: np.ndarray, feature: int, threshold: int) -> Tuple[EraShard, EraShard]:
    """Partition ``shard`` into left/right children using ``feature <= threshold``."""
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    for rows in shard.rows_per_era:
        if rows.size == 0:
            empty = rows[:0]
            left_rows.append(empty)
            right_rows.append(empty)
            continue
        bins = X_binned[rows, feature]
        mask = bins <= threshold
        if mask.all():
            left_rows.append(rows.astype(np.int32, copy=False))
            right_rows.append(_EMPTY)
            continue
        if (~mask).all():
            left_rows.append(_EMPTY)
            right_rows.append(rows.astype(np.int32, copy=False))
            continue
        left_rows.append(rows[mask])
        right_rows.append(rows[~mask])
    return EraShard.from_grouped_indices(left_rows), EraShard.from_grouped_indices(right_rows)
