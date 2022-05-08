from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline


@dataclass
class ParamGrid:
    """
    Adds hyperparameter key, value pairs to `param_grid` without pipeline prefixes.
    Currently only supports hyperparameters of the model and not the preprocessor!
    """

    param_grid: dict[str, Sequence] = field(default_factory=dict, init=False)

    def add_hyperparams(self, **kwargs) -> None:
        for key, value in kwargs.items():
            pipeline_key = "model__" + key
            self.param_grid[pipeline_key] = value


def cv_random(
    preprocessor: Pipeline,
    model: Any,
    param_grid: ParamGrid | None = None,
    n_folds: int = 10,
    random_state: int = 42,
    **hyperparams
):
    """
    Construct Randomized Cross Validation object from `Pipeline` components.
    Hyperparameters of the model can be added either by passing an existing `ParamGrid`
    object to the `param_grid` argument or as additional keyword arguments.
    """
    if param_grid is None:
        param_grid = ParamGrid()
    param_grid.add_hyperparams(**hyperparams)

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return HalvingRandomSearchCV(
        pipe,
        param_distributions=param_grid.param_grid,
        cv=n_folds,
        random_state=random_state,
    )
