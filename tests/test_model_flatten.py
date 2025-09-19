import numpy as np

from packboost.config import PackBoostConfig
from packboost.model import PackBoostModel, Tree, TreeNode


def test_flatten_forest_structure() -> None:
    config = PackBoostConfig()
    tree = Tree()
    root_id = tree.add_node(
        TreeNode(
            is_leaf=False,
            prediction=0.5,
            feature=1,
            threshold=2,
            left=1,
            right=2,
            depth=0,
        )
    )
    assert root_id == 0
    tree.add_node(TreeNode(is_leaf=True, prediction=1.5, depth=1))
    tree.add_node(TreeNode(is_leaf=True, prediction=-0.5, depth=1))

    model = PackBoostModel(
        config=config,
        bin_edges=None,
        initial_prediction=0.1,
        trees=[tree],
    )

    flattened = model.flatten_forest()

    np.testing.assert_array_equal(flattened.tree_offsets, np.array([0, 3], dtype=np.int32))
    np.testing.assert_array_equal(flattened.features, np.array([1, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(flattened.thresholds, np.array([2, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(flattened.lefts, np.array([1, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(flattened.rights, np.array([2, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(flattened.is_leaf, np.array([0, 1, 1], dtype=np.uint8))
    np.testing.assert_array_equal(flattened.values, np.array([0.5, 1.5, -0.5], dtype=np.float32))

    # Cached result should be reused
    assert model.flatten_forest() is flattened
