from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

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


class ClassificationMetrics(BaseMetrics):
    def __init__(
        self,
        y_true: np.ndarray,
        class_pred: np.ndarray,
        prob_pred: np.ndarray | None = None,
    ):
        super().__init__(y_true=y_true, class_pred=class_pred, prob_pred=prob_pred)

        self.accuracy = accuracy_score(self.y_true, self.class_pred)
        self.precision = precision_score(self.y_true, self.class_pred)
        self.recall = recall_score(self.y_true, self.class_pred)
        self.f1_score = f1_score(self.y_true, self.class_pred)

        self.auc = (
            roc_auc_score(self.y_true, self.prob_pred)
            if self.prob_pred is not None
            else roc_auc_score(self.y_true, self.class_pred)
        )

    def _to_dict(self) -> dict[str, float]:
        return {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1_score,
            "AUC": self.auc,
        }

    @property
    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.class_pred)

    def plot_confusion_matrix(self):
        ConfusionMatrixDisplay.from_predictions(self.y_true, self.class_pred)
        plt.show()


class RegressionMetrics(BaseMetrics):
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        super().__init__(y_true=y_true, y_pred=y_pred)

        self.mse = mean_squared_error(self.y_true, self.y_pred)
        self.mae = mean_absolute_error(self.y_true, self.y_pred)
        self.mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        self.r_squared = r2_score(self.y_true, self.y_pred)

    def _to_dict(self) -> dict[str, float]:
        return {
            "MSE": self.mse,
            "MAE": self.mae,
            "MAPE": self.mape,
            "R^2": self.r_squared,
        }


@dataclass
class MetricsComparison:
    metrics: Sequence[ClassificationMetrics]
    model_names: Sequence[str] | None = None

    def __post_init__(self):
        if self.model_names is not None and len(self.metrics) != len(self.model_names):
            raise ValueError("Lengths of `metrics` and `labels` must be identical.")

    def to_df(self) -> pd.DataFrame:
        return pd.concat([metric.to_df() for metric in self.metrics], axis=1).set_axis(
            self.model_names, axis=1
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

    def barplot(self, lower_bound: float | None = None) -> sns.FacetGrid:
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

        self._set_aesthetics(g, lower_bound)
        return g

    def stripplot(self, marker_size: int = 15) -> sns.FacetGrid:
        plot_df = self._setup_plot()

        g = sns.catplot(
            data=plot_df,
            x="score",
            y="index",
            kind="strip",
            col="metric",
            col_wrap=3,
            sharex=False,
            s=marker_size,
        )

        self._set_aesthetics(g, lower_bound=None)
        return g
