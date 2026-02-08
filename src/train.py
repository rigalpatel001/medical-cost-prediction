from sklearn.pipeline import Pipeline


def train_model(preprocessor, model, X_train, y_train):
    """
    Train a regression model using a preprocessing pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline.
    model :
        Regression model instance.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    Pipeline
        Trained model pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline
