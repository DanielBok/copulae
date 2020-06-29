from functools import lru_cache
from typing import Collection as C, Union

import numpy as np
from scipy.linalg import cholesky


class GMCParam:
    def __init__(self,
                 n_clusters: int,
                 n_dim: int,
                 prob: Union[C[float], np.ndarray],
                 means: Union[C[C[float]], np.ndarray],
                 covs: Union[C[C[C[float]]], np.ndarray]):
        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.prob = prob
        self.means = means
        self.covs = covs

    def to_vector(self):
        """
        Converts GMCParam to a 1-D vector representation. Vector representation enables GMCParam to be
        updated via scipy's optimize module.
        """

        def trans_upper_tri(x: np.ndarray, diag: bool):
            if diag:  # includes diagonal elements, thus do something about them
                x[np.diag_indices(self.n_dim)] = np.log(np.diag(x))
            return x[tri_indices(self.n_dim, diag)]

        scale = np.sqrt(np.diag(self.covs[0]))  # scale factor
        means = self.means / scale
        means = (means - means[0])[1:]

        # cholesky decomposition and rescaling
        vscale = scale[:, None] @ scale[None, :]  # variance scale factor
        covs = [cholesky(c) for c in (self.covs / vscale)]
        covs[0] = trans_upper_tri(covs[0], False)
        covs[1:] = [trans_upper_tri(c, True) for c in covs[1:]]

        # logit probs
        probs = np.log(self.prob / (1 - self.prob))

        return np.array([
            *probs,
            *means.ravel(),
            *np.ravel(covs[0]),
            *np.ravel(covs[1:])
        ])

    @classmethod
    def from_vector(cls, vector: Union[np.ndarray, C[float]], n_clusters: int, n_dim: int):
        """
        Converts a 1-D vector encoding to GMCParam

        Parameters
        ----------
        vector : np.ndarray
            Vector encoding which represents GMCParam
        n_clusters : int
            Number of components in copula mixture model
        n_dim : int
            Number of dimensions for each Gaussian component
        """
        m, d = n_clusters, n_dim
        vector = np.array(vector)
        prob = np.exp(vector[:m]) / (1 + np.exp(vector[:m]))

        means = np.zeros((m, d))
        means[1:] = vector[m:m + d * (m - 1)].reshape(m - 1, d)

        covs = []
        i = start = m + d * (m - 1)  # start of covariance index
        while i < len(vector):
            is_first = i == start
            if is_first:
                # indicator component, first sigma is the indicator component
                ind = tri_indices(d, False)  # upper triangle index
            else:
                ind = tri_indices(d, True)

            n = len(ind[0])
            um = np.zeros((d, d))  # upper matrix
            um[ind] = vector[i:i + n]

            if is_first:
                um[np.diag_indices(d)] = np.sqrt(1 - (um ** 2).sum(0))
            else:
                um[np.diag_indices(d)] = np.exp(np.diag(um))

            covs.append(um.T @ um)
            i += n

        return cls(m, d, prob, means, covs)

    @property
    def prob(self) -> np.ndarray:
        return self._prob

    @prob.setter
    def prob(self, value: Union[C[float], np.ndarray]):
        value = np.ravel(value)

        if len(value) != self.n_clusters:
            raise GMCParamError(f"probability vector should have {self.n_clusters} components")
        if any(value < 0) or any(value > 1) or not np.isclose(sum(value), 1):
            raise GMCParamError(f"Invalid values in probability vector")

        self._prob = value

    @property
    def means(self) -> np.ndarray:
        return self._means

    @means.setter
    def means(self, value: Union[C[float], np.ndarray]):
        value = np.asarray(value)

        shape = self.n_clusters, self.n_dim
        if value.shape != shape:
            raise GMCParamError(f"mean array should have shape {shape}")

        self._means = value

    @property
    def covs(self) -> np.ndarray:
        return self._covs

    @covs.setter
    def covs(self, value: Union[C[float], np.ndarray]):
        value = np.asarray(value)

        shape = self.n_clusters, self.n_dim, self.n_dim
        if value.shape != shape:
            raise GMCParamError(f"covariance array should have shape {shape}")

        self._covs = value

    def __str__(self):
        comps = []
        for i, (p, m, s) in enumerate(zip(self.prob, self.means, self.covs)):
            m = np.round(m, 4)

            cov_str = ""
            offset = ' ' * 15
            for j, x in enumerate(np.round(s, 4).tolist()):
                if j == 0:
                    cov_str += f"[{x},\n"
                elif j == self.n_dim - 1:
                    cov_str += f"{offset}{x}]"
                else:
                    cov_str += f"{offset}{x},\n"

            comps.append(f"""
Component   : {i + 1}
Probability : {round(p, 4)}
Mean        : {np.round(m, 4).tolist()}
Covariance  : {cov_str}
""".lstrip())
        sep = '\n' + '-' * 80 + '\n'
        comps = sep.join(comps)
        return f"<GMCParam\n{comps}/>"

    def __repr__(self):
        mean_repr = ",\n".join(f"{' ' * 8}{x.tolist()}" for x in self.means)

        covs_repr = ""
        offset = ' ' * 9
        for i, cov in enumerate(self.covs):  # type: int, np.ndarray
            for j, v in enumerate(cov.tolist()):
                if j == 0:
                    covs_repr += f"{offset[:-1]}[{v},\n"
                elif j == self.n_dim - 1:
                    covs_repr += f"{offset}{v}]"
                    if i < self.n_clusters - 1:
                        covs_repr += ',\n'
                else:
                    covs_repr += f"{offset}{v},\n"

        return f"""
GMCParam(
    n_clusters={self.n_clusters},
    n_dim={self.n_dim},
    prob={self.prob.tolist()},
    means=[
{mean_repr}
    ],
    covs=[
{covs_repr}
    ]
)""".strip()


@lru_cache(maxsize=128)
def tri_indices(dim: int, diag: bool):
    """Custom method to get upper triangle indices which goes column wise first"""
    r, c = [], []
    if diag:
        for i in range(dim):
            for j in range(dim):
                r.append(j)
                c.append(i)
                if i == j:
                    break

    else:
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    break
                r.append(j)
                c.append(i)

    return np.array(r), np.array(c)


class GMCParamError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
