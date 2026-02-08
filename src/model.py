from sklearn.ensemble import RandomForestRegressor
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


def get_random_forest_model(
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Create a Random Forest Regressor.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int | None
        Maximum depth of each tree.
    random_state : int
        Random seed.

    Returns
    -------
    RandomForestRegressor
        Random Forest regression model.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
