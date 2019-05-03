import os

import pandas as pd

__all__ = ["load_danube", "load_residuals"]

__module_path__ = os.path.dirname(__file__)


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

    Returns
    -------
    DataFrame
        A dataframe containing the Danube data
    """
    return pd.read_csv(os.path.join(__module_path__, 'data', 'danube.csv'))


def load_residuals() -> pd.DataFrame:
    """
    Loads a 394 x 7 array of regression residuals from unknown processes

    Returns
    -------
    DataFrame
        A data frame of simulated regression residuals
    """
    return pd.read_csv(os.path.join(__module_path__, 'data', 'residuals.csv'))
