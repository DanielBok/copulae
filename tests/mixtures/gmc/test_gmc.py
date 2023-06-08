from typing import Optional

import numpy as np
import pandas as pd
import pytest

from copulae.mixtures.gmc import EstimateMethod, GMCParam, GaussianMixtureCopula
from copulae.mixtures.gmc.estimators.em import Criteria
from copulae.mixtures.gmc.estimators.exceptions import InvalidStoppingCriteria
from copulae.mixtures.gmc.exception import GMCFitMethodError, GMCNotFittedError, GMCParamMismatchError
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
    return GaussianMixtureCopula(3, 2, param)


@pytest.fixture
def rvs(cop):
    np.random.seed(8)
    return cop.random(1000) + np.random.random((1000, 2))


def create_param(n_clusters: int, n_dim: int):
    p = np.random.random(n_clusters)
    p /= p.sum()

    covs = []
    for _ in range(n_clusters):
        cov = np.random.random((n_dim, n_dim))
        covs.append(cov @ cov.T)

    return GMCParam(n_clusters, n_dim,
                    prob=p,
                    means=np.random.random((n_clusters, n_dim)),
                    covs=covs)


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
        # ('pem', 'GMM'),
        # ('pem', 'Li'),
        ('sgd', None),
        ('kmeans', None),
    ])
    def test_fit(self, cop, rvs, method: EstimateMethod, criteria: Optional[Criteria]):
        # just testing the interface
        cop.fit(rvs, method=method, criteria=criteria)

    @pytest.mark.parametrize('method, criteria, error', [
        ('pem', 'bad-criteria', InvalidStoppingCriteria),
        ('bad-method', 'GMCM', GMCFitMethodError),
    ])
    def test_fit_with_bad_args(self, cop, rvs, method: EstimateMethod, criteria: Optional[Criteria], error: Exception):
        with pytest.raises(error):
            cop.fit(rvs, method=method, criteria=criteria)

    @pytest.mark.parametrize("bad_param, error", [
        ('bad param', GMCParamMismatchError),
        (create_param(10, 2), GMCParamMismatchError),
        (create_param(3, 10), GMCParamMismatchError),
        (create_param(10, 10), GMCParamMismatchError),
    ])
    def test_bad_param_raises_error(self, cop, bad_param, error):
        with pytest.raises(error):
            cop.params = bad_param

    def test_bounds(self, cop):
        # test that it does not raise error
        assert cop.bounds is NotImplemented

    @pytest.mark.parametrize("method", ['pdf', 'cdf'])
    def test_unfitted_copula_raises_error_on_method(self, method):
        cop = GaussianMixtureCopula(3, 2)

        with pytest.raises(GMCNotFittedError):
            getattr(cop, method)(data)
