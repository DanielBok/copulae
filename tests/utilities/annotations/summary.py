from importlib import import_module
from inspect import isabstract, isclass

import numpy as np
import pytest

import copulae
from copulae.copula.exceptions import NotFittedError

data = np.random.multivariate_normal(mean=[1, 2],
                                     cov=[[0.05, 0.08],
                                          [0.08, 0.13]],
                                     size=100)


def copulas():
    pkg = import_module('copulae')
    items = []
    for cop in (getattr(pkg, x) for x in dir(pkg)):
        if isclass(cop) and not isabstract(cop) and issubclass(cop, copulae.copula.BaseCopula):
            if issubclass(cop, copulae.EmpiricalCopula):
                items.append(cop(data))
            elif issubclass(cop, copulae.MarginalCopula):
                items.append(cop(copulae.NormalCopula(2), ['norm', 'norm']))
            elif issubclass(cop, copulae.GaussianMixtureCopula):
                items.append(cop(2, 2, {
                    'prob': [0.5, 0.5],
                    'means': [[0.1, 0.2],
                              [0.3, 0.4]],
                    'covs': [[[5, 6],
                              [6, 9]],
                             [[2, 3],
                              [3, 5]]]
                }))
            else:
                items.append(cop(2))

    return items


@pytest.mark.parametrize("cop", copulas())
def test_summary(cop):
    # haven't fit
    with pytest.raises(NotFittedError):
        cop.summary(category='fit')

    with pytest.raises(NotFittedError):
        cop.summary('fit')

    # # no errors
    cop.summary()
    cop.summary('copula')

    # after fitting, no errors
    cop.fit(data)
    cop.summary()
    cop.summary('fit')
    cop.summary('copula')

    with pytest.raises(ValueError):
        cop.summary("raises error")
