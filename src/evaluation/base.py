from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

sns.set_theme(style="whitegrid")


@dataclass
class BaseMetrics:
    y_true: np.ndarray
    y_pred: np.ndarray | None = None
    class_pred: np.ndarray | None = None
    prob_pred: np.ndarray | None = None

    def _to_dict(self) -> None:
        return NotImplementedError

    def to_dict(self) -> dict[str, float]:
        return self._to_dict()

    def to_df(self) -> pd.Series:
        return pd.Series(self.to_dict())

    def __repr__(self) -> str:
        return "\n".join(f"{key}: {value:.2f}" for key, value in self.to_dict().items())


@dataclass
class EvalClustering:
    model: KMeans | AgglomerativeClustering
    X: np.ndarray = field(repr=False)

    def __post_init__(self):
        if not self._is_fitted:
            self.model.fit(self.X)

        self.n_clusters: int = self.model.n_clusters
        self.labels: np.ndarray = self.model.labels_

    @property
    def _is_fitted(self) -> bool:
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False

    @staticmethod
    def _k_range_to_sequence(k_range: list[int] | tuple[int, int]) -> Sequence[int]:
        return range(k_range[0], k_range[1] + 1)

    @staticmethod
    def _plot_clusters(
        model: KMeans | AgglomerativeClustering,
        X: np.ndarray,
        ssr: float | None = None,
        linkage: str | None = None,
        ax: Any | None = None,
    ) -> None:
        """
        Implemented as static method such that it can be used both with instance
        attributes as inputs as well as custom inputs.
        """
        if ax is None:
            fig, ax = plt.subplots()

        svd = TruncatedSVD(2).fit(X)

        projections = svd.transform(X)
        ax.scatter(projections[:, 0], projections[:, 1], c=model.labels_)

        if isinstance(model, KMeans):
            projected_means = svd.transform(model.cluster_centers_)
            ax.scatter(
                projected_means[:, 0],
                projected_means[:, 1],
                color="black",
                marker="x",
                s=100,
            )

        if linkage is not None:
            add_title = f" | linkage = {linkage}"
        elif ssr is not None:
            add_title = f" | SSR = {ssr}"
        else:
            add_title = ""

        ax.set_title(
            f"{model.__class__.__name__}\n"
            f"n_clusters = {model.n_clusters}" + add_title
        )

    @staticmethod
    def _set_figdims(iterable: Sequence) -> tuple[int, int]:
        n_subplots = len(iterable)
        if n_subplots <= 3:
            nrows = 1
            ncols = n_subplots
        else:
            nrows = n_subplots // 3
            ncols = 3
        return nrows, ncols
