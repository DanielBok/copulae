from collections import abc
from typing import Optional

# noinspection PyProtectedMember
from scipy.stats._multivariate import multivariate_normal_gen as mng

from copulae.types import Numeric, OptNumeric


class multivariate_normal_gen(mng):

    def rvs(self, mean: OptNumeric = None, cov: Numeric = 1, size: Numeric = 1, random_state: Optional[int] = None):
        r = super().rvs(mean, cov, size, random_state)

        if not isinstance(size, int):
            dim = 1 if not isinstance(cov, abc.Sized) else len(cov)
            r = r.reshape(*size, dim)

        return r


multivariate_normal = multivariate_normal_gen()
