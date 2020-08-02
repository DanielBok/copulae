from pathlib import Path

import pandas as pd

__all__ = ["load_danube", "load_marginal_data", "load_residuals", "load_smi"]


def _load_file(fn: str) -> pd.DataFrame:
    path = Path(__file__).parent / "data" / fn
    return pd.read_csv(path, sep=r"\s*,", engine="python")


def load_danube() -> pd.DataFrame:
    """
    The danube dataset contains ranks of base flow observations from the Global River Discharge
    project of the Oak Ridge National Laboratory Distributed Active Archive Center (ORNL DAAC),
    a NASA data center. The measurements are monthly average flow rate for two stations situated
    at Scharding (Austria) on the Inn river and at Nagymaros (Hungary) on the Danube.

    The data have been pre-processed to remove any time trend. Specifically, Bacigal et al. (2011)
    extracted the raw data, and obtain the fast Fourier transformed centered observations. The
    negative spectrum is retained and a linear time series model with 12 seasonal components is
    fitted. Residuals are then extracted and AR model fitted to the series, the selection being
    done based on the AIC criterion with imposed maximum order of 3 and the number of autoregressive
    components may differ for each series.

    This data frame contains the following columns:

    inn:
        A numeric vector containing the rank of pre-whitened level observations of the Inn river
        at Nagyramos.

    donau:
        A numeric vector containing the rank of prewhitened level observations of the Donau river
        at Scharding.
    """
    return _load_file('danube.csv')


def load_marginal_data():
    """
    A simulated dataset where the marginal distributions are

    1. Student-T distribution (loc = 0, scale = 1, df = 16)
    2. Normal distribution (loc = 3, scale = 0.4)
    3. Exponential Distribution (scale = 0.5)

    Dependency structure (copula)
        Normal Copula with params [0.25, 0.4, 0.15]
    """
    return _load_file("marginal-data.csv")


def load_residuals():
    """
    Loads a 394 x 7 array of simulated regression residuals from unknown processes
    """
    return _load_file('residuals.csv')


def load_smi(as_returns=False) -> pd.DataFrame:
    """
    Dataset contains the close prices of all 20 constituents of the Swiss Market Index (SMI) from
    2011-09-09 to 2012-03-28.

    Parameters
    ----------
    as_returns: bool
        If true, transforms the price data to returns data
    """
    df = _load_file("smi.csv")
    df['DATE'] = pd.to_datetime(df.DATE)
    df.set_index("DATE", inplace=True)

    if as_returns:
        df = df.pct_change().dropna()

    return df
