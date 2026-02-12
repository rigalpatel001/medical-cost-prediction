import joblib


def save_model(model, path: str):
    """
    Save trained model pipeline to disk.
    """
    joblib.dump(model, path)


def load_model(path: str):
    """
    Load trained model pipeline from disk.
    """
    return joblib.load(path)
