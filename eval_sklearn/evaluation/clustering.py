from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans

from .base import EvalClustering

sns.set_theme(style="whitegrid")


class EvalKMeans(EvalClustering):
    def __init__(self, model, X):
        super().__init__(model=model, X=X)

        self.centers: np.ndarray = self.model.cluster_centers_
        self.ssr: float = self.model.inertia_

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={repr(self.model)}, "
            f"k={self.n_clusters}, ssr={np.round(self.ssr, 2)})"
        )

    def plot_clusters(self, ax: Any | None = None) -> None:
        self._plot_clusters(self.model, self.X, ssr=np.round(self.ssr, 2), ax=ax)

    def cross_validate(
        self,
        k_range: list[int] | tuple[int, int],
        with_plots: bool = True,
        fig_path: str | None = None,
    ) -> list[float]:
        ssr_list = []
        k_range = self._k_range_to_sequence(k_range)
        self._k_range = k_range

        if with_plots:
            nrows, ncols = self._set_figdims(k_range)
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
            )
            axes = axes.flat

        for i, k in enumerate(k_range):
            # ensure same model type as specified during initialization (KMeans or
            # Agglomerative) but with varying values for n_clusters
            model = clone(self.model)
            model.n_clusters = k

            ssr = model.fit(self.X).inertia_
            ssr_list.append(ssr)

            if with_plots:
                self._plot_clusters(model, self.X, ssr=np.round(ssr, 2), ax=axes[i])

        if fig_path is not None:
            fig.savefig(fig_path)

        self._ssr_list = ssr_list
        return ssr_list

    def plot_elbow(
        self,
        ax: Any | None = None,
        k_range: list[int] | tuple[int, int] | None = None,
        ssr_list: list[float] | None = None,
    ) -> None:
        """Elbow Plot for KMeans Algorithm to determine best number of clusters."""

        if not isinstance(self.model, KMeans):
            raise ValueError("Dendogram is only available for KMeans Clustering!")

        if ax is None:
            fig, ax = plt.subplots()

        if k_range is None:
            k_range = self._k_range
        else:
            k_range = self._k_range_to_sequence(k_range)

        # avoid refitting when cross_validate_k() was already used
        if ssr_list is None:
            ssr_list = self._ssr_list
        else:
            ssr_list = self.cross_validate(k_range)

        ax.plot(k_range, ssr_list, marker=".")
        ax.set_xticks(k_range)
        ax.set(
            title=f"Elbow Plot for {self.model.__class__.__name__}",
            xlabel="k",
            ylabel="SSR",
        )


class EvalAgglomerative(EvalClustering):
    def __init__(self, model, X):
        super().__init__(model=model, X=X)

        self.n_leaves: int = self.model.n_leaves_

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={repr(self.model)}, "
            f"n_clusters={self.n_clusters}, n_leaves={self.n_leaves})"
        )

    def plot_clusters(self, ax: Any | None = None) -> None:
        self._plot_clusters(
            self.model,
            self.X,
            linkage=self.model.linkage,
            ax=ax,
        )

    # adopted from scikit-learn documentation:
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    @staticmethod
    def _plot_dendrogram(model: AgglomerativeClustering, ax: Any | None = None) -> None:
        """Dendogram for Agglomerative Clustering to determine best number of clusters."""

        if not isinstance(model, AgglomerativeClustering):
            raise ValueError(
                "Dendogram is only available for Agglomerative Clustering!"
            )

        if ax is None:
            fig, ax = plt.subplots()

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, ax=ax)
        ax.set_title(
            f"Dendogram with {model.n_leaves_} leaves | Linkage = {model.linkage}"
        )

    def plot_dendrogram(self, ax: Any | None = None) -> None:
        self._plot_dendrogram(self.model, ax=ax)

    def _cross_validate_clusters(
        self,
        k_range: list[int] | tuple[int, int] = (2, 5),
        linkages: Sequence[str] = ("ward", "complete", "average", "single"),
    ):
        k_range = self._k_range_to_sequence(k_range)
        nrows = len(k_range)
        ncols = len(linkages)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
        )
        for i, k in enumerate(k_range):
            for j, linkage in enumerate(linkages):
                model = AgglomerativeClustering(
                    n_clusters=k, linkage=linkage, compute_distances=True
                )
                model.fit(self.X)

                self._plot_clusters(
                    model,
                    self.X,
                    linkage=linkage,
                    ax=axes[i, j],
                )

    def _cross_validate_dendogram(
        self,
        linkages: Sequence[str] = ("ward", "complete", "average", "single"),
    ):
        nrows, ncols = self._set_figdims(linkages)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
        )
        axes = axes.flat

        for i, linkage in enumerate(linkages):
            model = AgglomerativeClustering(
                n_clusters=2, linkage=linkage, compute_distances=True
            )
            model.fit(self.X)
            self._plot_dendrogram(model, ax=axes[i])

    def cross_validate(
        self,
        dendogram: bool = False,
        k_range: list[int] | tuple[int, int] = (2, 5),
        linkages: Sequence[str] = ("ward", "complete", "average", "single"),
    ) -> None:
        if dendogram:
            self._cross_validate_dendogram(linkages)
        else:
            self._cross_validate_clusters(k_range, linkages)
