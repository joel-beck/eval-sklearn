import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import make_blobs, make_classification, make_regression

from config import PublicConfig


def concat_targets_features(
    y: np.ndarray, X: np.ndarray, X_labels: list[str], target_col: str
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.Series(y, name=target_col),
            pd.DataFrame(X, columns=X_labels),
        ],
        axis=1,
    )


def main(testing: bool = False) -> None:
    load_dotenv()
    PROJECT_PATH = os.environ.get("PROJECT_PATH")
    assert PROJECT_PATH is not None
    data_dir = Path(PROJECT_PATH) / "data"

    file_ending = "testing.pkl" if testing else "notebooks.pkl"
    data_classification_path = Path(data_dir) / ("data_classification_" + file_ending)
    data_regression_path = Path(data_dir) / ("data_regression_" + file_ending)
    data_clustering_path = Path(data_dir) / ("data_clustering_" + file_ending)

    conf = PublicConfig()
    num_samples = conf.NUM_SAMPLES_TESTING if testing else conf.NUM_SAMPLES_NOTEBOOKS
    num_features = conf.NUM_FEATURES
    num_classification_targets = conf.NUM_TARGETS
    num_clusters = conf.NUM_CLUSTERS
    seed = conf.SEED
    target_col = conf.TARGET_COL

    X_labels = [f"x_{i}" for i in range(1, num_features + 1)]

    X_classification, y_classification = make_classification(
        num_samples,
        num_features,
        n_classes=num_classification_targets,
        random_state=seed,
    )
    data_classification = concat_targets_features(
        y_classification, X_classification, X_labels, target_col
    )

    X_regression, y_regression = make_regression(
        num_samples, num_features, random_state=seed
    )
    data_regression = concat_targets_features(
        y_regression, X_regression, X_labels, target_col
    )

    X_clustering, y_clustering = make_blobs(
        num_samples, num_features, centers=num_clusters, random_state=seed
    )
    data_clustering = concat_targets_features(
        y_clustering, X_clustering, X_labels, target_col
    )

    data_classification.to_pickle(data_classification_path)
    data_regression.to_pickle(data_regression_path)
    data_clustering.to_pickle(data_clustering_path)


if __name__ == "__main__":
    main(testing=False)
    main(testing=True)
