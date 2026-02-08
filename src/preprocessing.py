from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_column : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series
        Target variable.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
):
    """
    Split data into train and test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
