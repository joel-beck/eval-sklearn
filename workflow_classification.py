#%%
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from helpers_evaluation import ClassificationMetrics, MetricsComparison
from helpers_modeling import cv_random
from helpers_preprocessing import (
    get_column_transformer,
    get_feature_selector,
    get_preprocessor,
)

#%%
# SECTION: Configuration Parameters
DATA_PATH = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
TARGET_COL = "target"
SEED = 42

#%%
# SECTION: Setup Data
data = pd.read_csv(DATA_PATH)

# NOTE: For large number of missing values or small data set use Imputation Strategy
# https://scikit-learn.org/stable/modules/impute.html
all_rows = len(data)
data = data.dropna()
non_missing_rows = len(data)
print(f"Dropped {all_rows - non_missing_rows} rows with missing values.")
data.head()

#%%
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)
X_train.shape, X_test.shape

#%%
# SECTION: Preprocessing
column_transformer = get_column_transformer()
feature_selector = get_feature_selector(
    feature_selector="pca", pca_components=min(20, X_train.shape[1])
)
preprocessor = get_preprocessor(column_transformer, feature_selector)


#%%
# SECTION: Modeling
# SUBSECTION: Random Forest
random_forest = RandomForestClassifier(random_state=SEED)
cv_rf = cv_random(
    preprocessor, random_forest, random_state=SEED, n_estimators=range(10, 110, 10)
)
cv_rf.fit(X_train, y_train)

#%%
print(cv_rf.best_params_)

#%%
print(cv_rf.best_score_)

#%%
print(cv_rf.best_estimator_)

#%%
pd.DataFrame(cv_rf.cv_results_)

#%%
# SUBSECTION: HistGradientBoosting
gradient_boosting = HistGradientBoostingClassifier(random_state=SEED)
cv_gb = cv_random(
    preprocessor, gradient_boosting, random_state=SEED, max_depth=range(1, 6)
)
cv_gb.fit(X_train, y_train)

#%%
# SUBSECTION: XGBoost
xgboost = XGBClassifier(random_state=SEED)
cv_xgb = cv_random(preprocessor, xgboost, random_state=SEED, max_depth=range(1, 6))
cv_xgb.fit(X_train, y_train)

#%%
# SUBSECTION: LightGBM
lightgbm = LGBMClassifier(random_state=SEED)
cv_lgbm = cv_random(preprocessor, lightgbm, random_state=SEED, max_depth=range(1, 6))
cv_lgbm.fit(X_train, y_train)

#%%
# SECTION: Evaluation
# two ways to compute accuracy on test set:
# 1. cv.score(X_test, y_test)
# 2. accuracy_score(y_test, y_pred) with y_pred = cv.best_estimator_.predict(X_test)
for cv in [cv_rf, cv_gb, cv_xgb, cv_lgbm]:
    print(
        f"Accuracy {cv.best_estimator_['model'].__class__.__name__}: {cv.score(X_test, y_test):.2f}"
    )

#%%
# SUBSECTION: Overview of Metrics from individual models
y_pred_rf = cv_rf.best_estimator_.predict(X_test)
rf_metrics = ClassificationMetrics(y_test, y_pred_rf)
rf_metrics

y_pred_gb = cv_gb.best_estimator_.predict(X_test)
gb_metrics = ClassificationMetrics(y_test, y_pred_gb)
gb_metrics

y_pred_xgb = cv_xgb.best_estimator_.predict(X_test)
xgb_metrics = ClassificationMetrics(y_test, y_pred_xgb)
xgb_metrics

y_pred_lgbm = cv_lgbm.best_estimator_.predict(X_test)
lgbm_metrics = ClassificationMetrics(y_test, y_pred_lgbm)
lgbm_metrics

#%%
sns.set_theme(style="white")
print(rf_metrics.confusion_matrix)
rf_metrics.plot_confusion_matrix()
sns.set_theme(style="whitegrid")

#%%
# SUBSECTION: Compare Metrics from multiple different Models
metrics_comparison = MetricsComparison(
    metrics=[rf_metrics, gb_metrics, xgb_metrics, lgbm_metrics],
    labels=["Random Forest", "HistGradientBoosting", "XGBoost", "LightGBM"],
)
metrics_comparison.to_df()

#%%
metrics_comparison.barplot(lower_bound=0.7)

#%%
metrics_comparison.stripplot(marker_size=15)

#%%