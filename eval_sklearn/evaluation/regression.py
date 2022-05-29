import numpy as np
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .classification import BaseMetrics

sns.set_theme(style="whitegrid")


class RegressionMetrics(BaseMetrics):
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        super().__init__(y_true=y_true, y_pred=y_pred)

        self.mse = mean_squared_error(self.y_true, self.y_pred)
        self.mae = mean_absolute_error(self.y_true, self.y_pred)
        self.mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        self.r_squared = r2_score(self.y_true, self.y_pred)

    def to_dict(self) -> dict[str, float]:
        return {
            "MSE": self.mse,
            "MAE": self.mae,
            "MAPE": self.mape,
            "R^2": self.r_squared,
        }
