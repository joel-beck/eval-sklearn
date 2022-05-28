from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

from eval_sklearn.evaluation import EvalAgglomerative, EvalKMeans

# required to run pytest from package root directory, relative paths then refer to root
# directory rather than directory of test file
package_path = Path.cwd()

data: pd.DataFrame = pd.read_pickle(package_path / "data/data_clustering_testing.pkl")

config_public = ConfigParser(interpolation=ExtendedInterpolation())
config_public.read(package_path / "config_public.ini")
seed = config_public.getint("Constants", "seed")
target_col = config_public.get("Names", "target_col")

X = data.drop(columns=target_col)

kmeans = KMeans(n_clusters=3, random_state=seed)
eval_kmeans = EvalKMeans(kmeans, X)

agg = AgglomerativeClustering(n_clusters=5, compute_distances=True, linkage="ward")
eval_agg = EvalAgglomerative(agg, X)


def test_initialization_kmeans():
    assert eval_kmeans._is_fitted
    assert eval_kmeans.n_clusters == 3
    assert np.unique(eval_kmeans.labels).tolist() == [0, 1, 2]
    assert eval_kmeans.ssr > 0
    assert eval_kmeans.centers.shape == (3, 20)


def test_cross_validation_kmeans():
    _ = eval_kmeans.cross_validate(k_bounds=(1, 6), with_plots=False)
    _ = eval_kmeans.cross_validate(k_bounds=(1, 6), with_plots=True)
    assert len(eval_kmeans._ssr_list) == 6
    assert eval_kmeans._ssr_list == sorted(eval_kmeans._ssr_list, reverse=True)
    assert eval_kmeans._k_range == range(1, 7)


def test_plotting_kmeans():
    eval_kmeans.plot_clusters()
    eval_kmeans.plot_elbow()


def test_initialization_agglomerative():
    assert eval_agg._is_fitted
    assert eval_agg.n_clusters == 5
    assert np.unique(eval_agg.labels).tolist() == [0, 1, 2, 3, 4]
    assert eval_agg.n_leaves > 0


def test_cross_validation_agglomerative():
    eval_agg.cross_validate(dendogram=False, k_range=(1, 5))
    eval_agg.cross_validate(dendogram=True, linkages=["ward", "complete", "single"])


def test_plotting_agglomerative():
    eval_agg.plot_clusters()
    eval_agg.plot_dendrogram()
