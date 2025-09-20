from packboost.config import PackBoostConfig


def test_config_defaults():
    cfg = PackBoostConfig()
    assert cfg.pack_size == 8
    assert cfg.max_depth == 6
    assert cfg.device == "cpu"
    assert cfg.max_bins <= 256
