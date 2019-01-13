import pytest
import pandas as pd


@pytest.fixture(scope='session')
def residual_data():
    df = pd.read_csv('data/residuals.csv')
    return df.values
