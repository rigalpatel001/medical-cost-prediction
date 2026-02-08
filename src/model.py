from sklearn.linear_model import LinearRegression


def get_linear_regression_model() -> LinearRegression:
    """
    Create a Linear Regression model.

    Returns
    -------
    LinearRegression
        Scikit-learn Linear Regression model.
    """
    return LinearRegression()
