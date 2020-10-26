import numpy as np
import pandas as pd
import pytest

from copulae import EmpiricalCopula
from copulae.datasets import load_marginal_data


@pytest.fixture(scope="module")
def copula():
    data = load_marginal_data()
    return EmpiricalCopula(data)


def test_pandas_dataframe_outputs(copula):
    output = copula.random(10)
    assert isinstance(output, pd.DataFrame) and output.shape == (10, copula.dim)


@pytest.mark.parametrize("inputs, output_cls", [
    # 1D inputs get cast to 2D and then gets returned as float
    ([0.1, 0.2, 0.3], float),
    ([[0.1, 0.2, 0.3]], float),
    (pd.Series([0.1, 0.2, 0.3]), float),
    (pd.Series({"STUDENT": 0.1, "EXP": 0.2, "NORM": 0.3}), float),
    ([[0.1, 0.2, 0.3],
      [0.2, 0.3, 0.4]
      ], np.ndarray),
    (np.array([[0.1, 0.2, 0.3],
               [0.2, 0.3, 0.4]
               ]), np.ndarray),
    (pd.DataFrame({
        "EXP": [0.2, 0.3],
        "NORM": [0.2, 0.3],
        "STUDENT": [0.2, 0.3],
    }), np.ndarray),
])
def test_series_inputs(copula, inputs, output_cls):
    output = copula.cdf(inputs)
    assert isinstance(output, output_cls)
