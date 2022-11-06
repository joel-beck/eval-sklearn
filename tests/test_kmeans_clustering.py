import numpy as np
import pytest
from sklearn.cluster import KMeans

from eval_sklearn.evaluation import EvalKMeans


@pytest.fixture
def eval_kmeans(config, data) -> EvalKMeans:
    kmeans = KMeans(n_clusters=3, random_state=config.SEED)
    return EvalKMeans(kmeans, data)


def test_initialization_kmeans(eval_kmeans) -> None:
    assert eval_kmeans._is_fitted
    assert eval_kmeans.n_clusters == 3
    assert np.unique(eval_kmeans.labels).tolist() == [0, 1, 2]
    assert eval_kmeans.ssr > 0
    assert eval_kmeans.centers.shape == (3, 20)


def test_cross_validation_kmeans(eval_kmeans) -> None:
    _ = eval_kmeans.cross_validate(k_bounds=(1, 6), with_plots=False)
    _ = eval_kmeans.cross_validate(k_bounds=(1, 6), with_plots=True)
    assert len(eval_kmeans._ssr_list) == 6
    assert eval_kmeans._ssr_list == sorted(eval_kmeans._ssr_list, reverse=True)
    assert eval_kmeans._k_range == range(1, 7)


def test_plotting_kmeans(eval_kmeans) -> None:
    _ = eval_kmeans.cross_validate(k_bounds=(1, 6), with_plots=False)
    eval_kmeans.plot_clusters()
    eval_kmeans.plot_elbow()
