"""Core data structures and algorithms for PackBoost FAST PATH."""

from .dataset import group_rows_by_era
from .frontier import evaluate_frontier_fastpath
from .shards import EraShard, FrontierDecision, NodeState, split_shard

__all__ = [
    "EraShard",
    "FrontierDecision",
    "NodeState",
    "group_rows_by_era",
    "evaluate_frontier_fastpath",
    "split_shard",
]
