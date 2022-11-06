import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering

from eval_sklearn.evaluation import EvalAgglomerative


@pytest.fixture
def eval_agg(data) -> EvalAgglomerative:
    agg = AgglomerativeClustering(n_clusters=5, compute_distances=True, linkage="ward")
    return EvalAgglomerative(agg, data)


def test_initialization_agglomerative(eval_agg) -> None:
    assert eval_agg._is_fitted
    assert eval_agg.n_clusters == 5
    assert np.unique(eval_agg.labels).tolist() == [0, 1, 2, 3, 4]
    assert eval_agg.n_leaves > 0


def test_cross_validation_agglomerative(eval_agg) -> None:
    eval_agg.cross_validate(dendogram=False, k_range=(1, 5))
    eval_agg.cross_validate(dendogram=True, linkages=["ward", "complete", "single"])


def test_plotting_agglomerative(eval_agg) -> None:
    eval_agg.plot_clusters()
    eval_agg.plot_dendrogram()
