from collections.abc import Sequence
from typing import Any

import seaborn as sns
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline

sns.set_theme(style="whitegrid")


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
