from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate regression model performance.

    Returns
    -------
    dict
        Dictionary containing RMSE and R2 score.
    """
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "RMSE": rmse,
        "R2": r2,
    }
