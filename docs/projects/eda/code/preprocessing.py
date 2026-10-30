"""Pipeline importável: nenhuma estatística é calculada na construção."""

import numpy as np
from dataset import CATEGORICAL, ENGAGEMENT, RATES
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def make_preprocessor() -> Pipeline:
    """Ajustar com fit(X_train); teste recebe exclusivamente transform(X_test).

    Recebe DataFrame com os nomes originais. Revenue e PageValues, caso sejam
    fornecidos, são ignorados por remainder='drop'. Códigos categóricos devem
    estar em strings, como em dataset.load_data().
    """
    engagement = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    rates = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="Ausente", keep_empty_features=True),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
            ),
        ]
    )
    columns = ColumnTransformer(
        [
            ("engagement", engagement, ENGAGEMENT),
            ("rates", rates, RATES),
            ("category", categorical, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("columns", columns)])
