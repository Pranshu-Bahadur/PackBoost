"""PackBoost: pack-parallel gradient boosting with DES scoring."""

from .booster import PackBoost
from .config import PackBoostConfig

__all__ = ["PackBoost", "PackBoostConfig"]
