from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


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


def main():
    config_private = ConfigParser(interpolation=ExtendedInterpolation())
    config_private.read("config_private.ini")

    data_dir = config_private.get("Paths", "data_dir")
    data_classification_path = Path(data_dir) / config_private.get(
        "Paths", "filename_classification"
    )
    data_regression_path = Path(data_dir) / config_private.get(
        "Paths", "filename_regression"
    )

    config_public = ConfigParser(interpolation=ExtendedInterpolation())
    config_public.read("config_public.ini")

    num_samples = config_public.getint("Constants", "num_samples")
    num_features = config_public.getint("Constants", "num_features")
    num_classification_targets = config_public.getint(
        "Constants", "num_classification_targets"
    )
    seed = config_public.getint("Constants", "seed")

    target_col = config_public.get("Names", "target_col")
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

    data_classification.to_pickle(data_classification_path)
    data_regression.to_pickle(data_regression_path)


if __name__ == "__main__":
    main()
