import sys

import numpy as np
from sklearn.cluster import KMeans

from copulae.core import pseudo_obs
from copulae.mixtures.gmc.parameter import GMCParam
from .summary import FitSummary

if sys.platform == 'win32':
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        del patch_sklearn
    except ImportError:
        pass


def k_means(data: np.ndarray, n_clusters: int, n_dim: int, ties='average'):
    """
    Determines the GMC's parameters via K-means

    Parameters
    ----------
    data : np.ndarray
        Input data

    n_clusters : int
        Number of clusters (components)

    n_dim : int
        Number of dimension for each Gaussian distribution

    ties : { 'average', 'min', 'max', 'dense', 'ordinal' }, optional
        Specifies how ranks should be computed if there are ties in any of the coordinate samples. This is
        effective only if the data has not been converted to its pseudo observations form
    """
    u = pseudo_obs(data, ties)
    km = KMeans(n_clusters)
    km.fit(u)

    groups, prob = np.unique(km.labels_, return_counts=True)
    prob = prob / sum(prob)

    means = np.array([data[km.labels_ == g].mean(0) for g in groups])
    covs = np.array([np.cov(data[km.labels_ == g], rowvar=False) for g in groups])

    return FitSummary(GMCParam(n_clusters, n_dim, prob, means, covs), True, 'kmeans', len(u),
                      {'Inertia': km.inertia_, 'ties': ties})
