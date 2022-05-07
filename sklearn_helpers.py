from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sns.set_theme(style="whitegrid")
from collections.abc import Sequence


@dataclass
class ClassificationMetrics:
    y_true: np.ndarray
    class_pred: np.ndarray
    prob_pred: np.ndarray | None = None

    def __post_init__(self):
        self.accuracy = accuracy_score(self.y_true, self.class_pred)
        self.precision = precision_score(self.y_true, self.class_pred)
        self.recall = recall_score(self.y_true, self.class_pred)
        self.f1_score = f1_score(self.y_true, self.class_pred)

        self.auc = (
            roc_auc_score(self.y_true, self.prob_pred)
            if self.prob_pred is not None
            else roc_auc_score(self.y_true, self.class_pred)
        )

    def __repr__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.2f}\n"
            f"Precision: {self.precision:.2f}\n"
            f"Recall: {self.recall:.2f}\n"
            f"F1 Score: {self.f1_score:.2f}\n"
            f"AUC: {self.auc:.2f}"
        )

    def to_dict(self) -> dict[str, float]:
        return dict(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f1_score=self.f1_score,
            auc=self.auc,
        )

    def to_df(self) -> pd.Series:
        return pd.Series(self.to_dict())


@dataclass
class MetricsComparison:
    metrics: Sequence[ClassificationMetrics]
    labels: Sequence[str] | None = None
    lower_bound: float | None = None
    marker_size: int | None = None

    def __post_init__(self):
        if self.labels is not None and len(self.metrics) != len(self.labels):
            raise ValueError("Lengths of `metrics` and `labels` must be identical.")

    def to_df(self) -> pd.DataFrame:
        return pd.concat([metric.to_df() for metric in self.metrics], axis=1).set_axis(
            self.labels, axis=1
        )

    def _setup_plot(self) -> pd.DataFrame:
        metrics_label_mapping = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score",
            "auc": "AUC",
        }

        plot_df = (
            self.to_df()
            .T.melt(var_name="metric", value_name="score", ignore_index=False)
            .reset_index()
            .assign(metric=lambda x: x["metric"].replace(metrics_label_mapping))
        )

        return plot_df

    def _set_aesthetics(self, g: sns.FacetGrid, lower_bound: int | None) -> None:
        g.set_titles(col_template="{col_name}").set_axis_labels(x_var="", y_var="").set(
            xlim=(lower_bound, None)
        )

    def barplot(self) -> sns.FacetGrid:
        plot_df = self._setup_plot()

        g = sns.catplot(
            data=plot_df,
            x="score",
            y="index",
            kind="bar",
            col="metric",
            col_wrap=3,
            sharex=False,
        )

        self._set_aesthetics(g, self.lower_bound)
        return g

    def stripplot(self) -> sns.FacetGrid:
        plot_df = self._setup_plot()

        g = sns.catplot(
            data=plot_df,
            x="score",
            y="index",
            kind="strip",
            col="metric",
            col_wrap=3,
            sharex=False,
            s=self.marker_size,
        )

        self._set_aesthetics(g, lower_bound=None)
        return g
