from sklearn.pipeline import Pipeline

from src.model import get_linear_regression_model


def train_model(preprocessor, X_train, y_train):
    """
    Train a regression model using a preprocessing pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    Pipeline
        Trained model pipeline.
    """
    model = get_linear_regression_model()

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline
