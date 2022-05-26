import numpy as np

from eval_sklearn.evaluation import ClassificationMetrics


def test_default():
    assert 1 + 1 == 2


def test_basic_imports():
    y_true = np.random.choice(a=[0, 1], size=10)
    class_pred = np.random.choice(a=[0, 1], size=10)
    classification_metrics = ClassificationMetrics(y_true, class_pred)

    assert classification_metrics.__class__.__name__ == "ClassificationMetrics"
