import numpy as np
import pandas as pd
import pytest

from copulae.mixtures.gmc import GaussianMixtureCopula
from copulae.mixtures.gmc.estimators.exceptions import InvalidStoppingCriteria
from copulae.mixtures.gmc.exception import GMCFitMethodError
from copulae.mixtures.gmc.summary import Summary
from .common import param

data = pd.DataFrame([
    [-25.6358480196781, -7.68033291597402],
    [-25.7920453752756, -2.12280740853803],
    [-22.6543385982607, -2.87536240104291],
    [-9.72879108218027, -11.1520025098258],
    [-24.3045290123645, -6.1431164444914],
    [-24.6518826480268, -7.23850007076786],
], columns=["C1", "C2"])


@pytest.fixture
def cop():
    c = GaussianMixtureCopula(3, 2)
    c.params = param

    return c


@pytest.fixture
def rvs(cop):
    return cop.random(1000) + np.random.random((1000, 2))


class TestGaussianMixtureCopula:
    def test_pdf(self, cop):
        assert isinstance(cop.pdf(data), np.ndarray)

    def test_cdf(self, cop):
        assert isinstance(cop.cdf(data), np.ndarray)

    def test_random(self, cop):
        rvs = cop.random(1000)
        assert rvs.shape == (1000, 2)

    def test_summary(self, cop):
        assert isinstance(cop.summary(), Summary)

    @pytest.mark.parametrize('method, criteria', [
        ('pem', 'GMCM'),
        ('pem', 'GMM'),
        ('pem', 'Li'),
        ('sgd', None),
        ('kmeans', None),
    ])
    def test_fit(self, cop, rvs, method, criteria):
        # just testing the interface
        cop.fit(rvs, method=method, criteria=criteria)

    @pytest.mark.parametrize('method, criteria, error', [
        ('pem', 'bad-criteria', InvalidStoppingCriteria),
        ('bad-method', 'GMCM', GMCFitMethodError),
    ])
    def test_fit_with_bad_args(self, cop, rvs, method, criteria, error):
        with pytest.raises(error):
            cop.fit(rvs, method=method, criteria=criteria)
