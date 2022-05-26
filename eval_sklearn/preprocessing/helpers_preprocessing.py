import seaborn as sns
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")


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
