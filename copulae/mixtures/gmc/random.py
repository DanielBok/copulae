from typing import Collection as C, Tuple, Union

import numpy as np
from numpy.random import choice
from scipy.stats import multivariate_normal as mvn

__all__ = ['random_gmcm']


def random_gmcm(n: int, m: int, d: int,
                prob: Union[C[float], np.ndarray],
                means: Union[C[C[float]], np.ndarray],
                covs: Union[C[C[C[float]]], np.ndarray]):
    prob, means, cov = validate(prob, means, covs, m, d)
    rand = random_gmm(n, m, d, prob, means, covs)
    '''latent realizations from Gaussian mixture model'''

    out = np.zeros_like(rand)
    for mus, sigmas, p in zip(means, covs, prob):  # type: np.ndarray, np.ndarray, float
        for j in range(d):
            m = mus[j]
            s = sigmas[j, j] ** 0.5
            out[:, j] = out[:, j] + p * approximate_pnorm(rand[:, j], m, s)

    return out


def random_gmm(n: int, m: int, d: int,
               prob: Union[C[float], np.ndarray],
               means: Union[C[C[float]], np.ndarray],
               covs: Union[C[C[C[float]]], np.ndarray]):
    prob, means, covs = validate(prob, means, covs, m, d)

    output = np.empty((n, d))
    order = choice(range(m), n, p=prob)
    for i in range(m):
        k = sum(order == i)
        output[order == i] = mvn.rvs(means[i], cov=covs[i], size=k)

    return output


def validate(prob, means, covs, m: int, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = np.array(prob)
    assert len(prob) == m, f"should have {m} elements in mixture probability "
    assert all(prob > 0) and np.isclose(prob.sum(), 1), "mixture probabilities are invalid"

    means = np.array(means)
    assert means.size == (m, d), f"mean arrays should have size {(m, d)}"

    covs = np.array(covs)
    assert covs.size == (m, d, d), f"covariance arrays should have size {(m, d, d)}"

    return prob, means, covs


def approximate_pnorm(vec: np.ndarray, mu: float, sd: float):
    """
    Approximate univariate Gaussian CDF, applid marginally
    Abramowitz, Stegun p. 299 (7.1.25) (using error function) improved.
    """
    a1 = 0.3480242
    a2 = -0.0958798
    a3 = 0.7478556
    p = 0.47047
    sqrt2 = 1.4142136

    z = (vec - mu) / (sd * sqrt2)
    zi = np.abs(z)
    t = 1 / (1 + p * zi)
    chunk = 0.5 * (a1 * t + a2 * t ** 2 + a3 * t ** 3) * np.exp(-(zi ** 2))

    return np.where(z < 0, chunk, 1 - chunk)


rand = np.array([
    [-1.2903482, 11.157217],
    [-6.8580951, -13.681541],
    [-1.9675101, 5.626113],
    [-0.7410138, 10.641491],
    [-8.2591179, -16.009588],
    [-0.9330397, 9.224378],
    [-1.4539036, 8.523645],
    [-0.9358209, 6.602879],
    [-2.8088686, 14.350002],
    [-6.2604705, -15.381642],
    [-3.0214262, 8.485776],
    [-1.9872259, 10.599498],
    [-1.2634725, 10.179220],
    [-15.0439012, -24.957682],
    [-0.5047083, 9.022975],
    [-5.6959585, -15.115005],
    [-1.3788111, 11.408607],
    [-6.8501888, -15.162949],
    [-2.6976115, 9.217961],
    [-1.3201052, 10.732456],
    [-5.7857012, -15.383855],
    [-1.9326496, 9.893376],
    [-7.4390242, -15.644855],
    [-1.6159976, 8.037682],
    [-2.6665944, 10.865855],
    [-19.2804765, -28.194438],
    [-1.9121354, 10.061137],
    [-5.2117480, -14.862173],
    [-0.7502507, 9.090076],
    [-0.8535153, 10.013243],
])

d = 2
prob = [0.04400228, 0.31716375, 0.63883398]
means = np.array([
    [-16.60637, -26.62158],
    [-6.565751, -14.501430],
    [-1.589503, 10.240296],
])

covs = np.array([
    [[1.777549, 1.040045],
     [1.040045, 1.346215]],
    [[0.5104765, 0.2927911],
     [0.2927911, 1.6621114]],
    [[0.5941591, -0.3785594],
     [-0.3785594, 2.7328910]]
])
