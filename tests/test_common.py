"""
Placing common copula method/function tests here
"""
import numpy as np
import pandas as pd
import pytest

from copulae import GaussianCopula
from copulae.copula.exceptions import InputDataError


def faulty_data():
    return pd.DataFrame({
        'A': [-1, 0.5, 1],
        'B': [-1, 0.5, 1],
    })


@pytest.mark.parametrize("data", [
    faulty_data(),  # not marginals
    faulty_data().to_numpy(),  # not marginals
    faulty_data().to_numpy().tolist(),  # not marginals
    np.random.uniform(size=4),  # wrong shape (ndim = 1, should be 2)
    np.random.uniform(size=(2, 3, 4)),  # wrong shape (ndim = 3, should be 2)
    np.random.uniform(size=(5, 3))  # should only have 2 columns, but has 3
])
def test_fit_errors(data):
    cop = GaussianCopula(2)

    with pytest.raises(InputDataError):
        cop.fit(data, to_pobs=False)
