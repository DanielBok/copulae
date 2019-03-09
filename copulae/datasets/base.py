import os

import pandas as pd

__all__ = ["load_residuals"]

__module_path__ = os.path.dirname(__file__)


def load_residuals() -> pd.DataFrame:
    """
    Loads a 394 x 7 array of regression residuals from unknown processes

    Returns
    -------
    DataFrame
        A data frame of simulated regression residuals
    """

    return pd.read_csv(os.path.join(__module_path__, 'data', 'residuals.csv'))
