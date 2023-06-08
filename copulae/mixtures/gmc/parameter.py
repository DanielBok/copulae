from typing import Collection as C, Iterable, Tuple, TypedDict, Union

import numpy as np

from copulae.core import near_psd
from copulae.mixtures.gmc.exception import GMCParamError

__all__ = ['GMCParam', 'GMCParamDict']


class GMCParamDict(TypedDict):
    prob: Union[C[float], np.ndarray]
    means: Union[C[C[float]], np.ndarray]
    covs: Union[C[C[C[float]]], np.ndarray]


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

    @classmethod
    def from_dict(cls, param: GMCParamDict):
        means = np.asarray(param['means'])
        n_clusters, n_dim = means.shape
        return cls(n_clusters, n_dim, np.asarray(param['prob']), means, np.asarray(param['covs']))

    def to_vector(self):
        """
        Converts GMCParam to a 1-D vector representation. Vector representation enables GMCParam to be
        updated via scipy's optimize module.
        """
        self.prob[self.prob == 0] += 1e-5
        self.prob[self.prob == 1] -= 1e-5
        self.prob /= self.prob.sum()

        return np.array([
            *np.log(self.prob / (1 - self.prob)),
            *self.means.ravel(),
            *np.ravel([c[np.triu_indices(self.n_dim)] for c in self.covs])
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
        prob /= prob.sum()

        means = vector[m:(m + m * d)].reshape(m, d)

        covs = []
        step = d * (d + 1) // 2
        for i in range(m * (d + 1), len(vector), step):
            cov = np.zeros((d, d))
            cov[np.triu_indices(d)] = vector[i:i + step]
            cov = np.triu(cov, 1) + cov.T
            covs.append(cov)

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

        self._covs = np.array([near_psd(cov) for cov in value])  # forces covariance matrix to be psd

    def __iter__(self) -> Iterable[Tuple[float, Union[float, np.ndarray], np.ndarray]]:
        for i in range(self.n_clusters):
            yield self.prob[i], self.means[i], self.covs[i]

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
