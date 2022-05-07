from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")
from collections.abc import Sequence


def get_column_transformer() -> ColumnTransformer:
    """
    Returns ColumnTransformer Object which standardizes all numeric Variables and
    transforms all categorical Variables to Dummy Variables with entries 0 and 1.
    """
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        [
            ("scaler", scaler, make_column_selector(dtype_include="number")),
            ("encoder", encoder, make_column_selector(dtype_include="object")),
        ],
    )


def get_feature_selector(
    feature_selector: str,
    pca_components: int | None = None,
    k: int = 10,
) -> PCA | SelectKBest:
    """
    Returns either a PCA or a SelectKBest Object. The number of resulting dimensions
    after application can be specified with input parameters.
    """

    feature_selectors = {"pca": PCA(pca_components), "k_best": SelectKBest(k=k)}
    return feature_selectors[feature_selector]


def get_preprocessor(
    column_transformer: ColumnTransformer, feature_selector: PCA | SelectKBest
) -> Pipeline:
    """
    Creates Pipeline Object that first standardizes all numeric Variables and encodes
    categorical Variables as Dummy Variables and then reduces the Dimensionality of the
    Feature Space.
    """

    return Pipeline(
        [
            ("column_transformer", column_transformer),
            ("feature_selector", feature_selector),
        ]
    )


def setup_cv(
    preprocessor: Pipeline,
    model: Any,
    param_grid: dict | None = None,
    pipeline_keys: Sequence[str] | None = None,
    random_state: int = 42,
):
    if pipeline_keys is None:
        pipeline_keys = ["preprocessor", "model"]

    pipe = Pipeline([(pipeline_keys[0], preprocessor), (pipeline_keys[1], model)])
    return HalvingRandomSearchCV(
        pipe, param_distributions=param_grid, random_state=random_state
    )


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
