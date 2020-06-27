from typing import Collection as C, Union

import numpy as np


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


class GMCParamError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
