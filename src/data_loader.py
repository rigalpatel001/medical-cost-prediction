from pathlib import Path
import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the medical cost dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df
