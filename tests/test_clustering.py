from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

from config import PublicConfig
from eval_sklearn.evaluation import EvalAgglomerative, EvalKMeans


# NOTE: dotenv package FAILS in CI pipeline without mocking
# required to run pytest from package root directory, relative paths then refer to root
# directory rather than directory of test file
@pytest.fixture
def root_path() -> Path:
    return Path.cwd()


@pytest.fixture
def config() -> PublicConfig:
    return PublicConfig()


@pytest.fixture
def data(root_path, config) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_pickle(
        root_path / "data" / "data_clustering_testing.pkl"
    )
    return data.drop(columns=config.TARGET_COL)


@pytest.fixture
def eval_kmeans(config, data) -> EvalKMeans:
    kmeans = KMeans(n_clusters=3, random_state=config.SEED)
    return EvalKMeans(kmeans, data)


@pytest.fixture
def eval_agg(data) -> EvalAgglomerative:
    agg = AgglomerativeClustering(n_clusters=5, compute_distances=True, linkage="ward")
    return EvalAgglomerative(agg, data)


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
