import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from copulae.stats import corr, pearson_rho, spearman_rho, kendall_tau

DP = 3


@pytest.fixture
def data():
    np.random.seed(8888)
    return np.random.uniform(-10, 10, size=(30, 3))


def test_corr_pearson(data: np.array):
    assert_array_almost_equal(pearson_rho(data),
                              np.array([[1.0000000, -0.16871767, -0.23407923],
                                        [-0.1687177, 1.00000000, 0.05121912],
                                        [-0.2340792, 0.05121912, 1.00000000]]),
                              DP)

    data[-1, 0] = np.nan

    with np.errstate(invalid='ignore'):
        # Adding ignore because comparing nans raise invalid runtime warning
        assert_array_almost_equal(pearson_rho(data),
                                  np.array([[1.0000000, np.nan, np.nan],
                                            [np.nan, 1.00000000, 0.05121912],
                                            [np.nan, 0.05121912, 1.00000000]]),
                                  DP)

    assert_array_almost_equal(pearson_rho(data, use='complete'),
                              np.array([[1.0000000, -0.16634284, -0.30256565],
                                        [-0.16634284, 1.00000000, 0.06097189],
                                        [-0.30256565, 0.06097189, 1.00000000]]),
                              DP)

    assert_array_almost_equal(pearson_rho(data, use='pairwise.complete'),
                              np.array([[1.0000000, -0.16634284, -0.30256565],
                                        [-0.16634284, 1.00000000, 0.05121912],
                                        [-0.30256565, 0.05121912, 1.00000000]]),
                              DP)


def test_corr_kendall(data: np.array):
    assert_array_almost_equal(kendall_tau(data),
                              np.array([[1.0000000, -0.02988506, -0.18160920],
                                        [-0.02988506, 1.00000000, 0.02068966],
                                        [-0.18160920, 0.02068966, 1.00000000]]),
                              DP)

    data[-1, 0] = np.nan

    assert_array_almost_equal(kendall_tau(data),
                              np.array([[1.0000000, np.nan, np.nan],
                                        [np.nan, 1.00000000, 0.02068966],
                                        [np.nan, 0.02068966, 1.00000000]]),
                              DP)

    assert_array_almost_equal(kendall_tau(data, use='complete'),
                              np.array([[1.0000000, -0.03940887, -0.23152709],
                                        [-0.03940887, 1.00000000, 0.02955665],
                                        [-0.23152709, 0.02955665, 1.00000000]]),
                              DP)

    assert_array_almost_equal(kendall_tau(data, use='pairwise.complete'),
                              np.array([[1.0000000, -0.03940887, -0.23152709],
                                        [-0.03940887, 1.00000000, 0.02068966],
                                        [-0.23152709, 0.02068966, 1.00000000]]),
                              DP)


def test_corr_spearman(data: np.array):
    assert_array_almost_equal(spearman_rho(data),
                              np.array([[1.00000000, -0.03136819, -0.2520578],
                                        [-0.03136819, 1.00000000, 0.0246941],
                                        [-0.25205784, 0.02469410, 1.0000000]]),
                              DP)

    data[-1, 0] = np.nan

    assert_array_almost_equal(spearman_rho(data),
                              np.array([[1.0000000, np.nan, np.nan],
                                        [np.nan, 1.00000000, 0.0246941],
                                        [np.nan, 0.0246941, 1.00000000]]),
                              DP)

    assert_array_almost_equal(spearman_rho(data, use='complete'),
                              np.array([[1.0000000, -0.03793103, -0.31527094],
                                        [-0.03793103, 1.00000000, 0.03103448],
                                        [-0.31527094, 0.03103448, 1.00000000]]),
                              DP)

    assert_array_almost_equal(spearman_rho(data, use='pairwise.complete'),
                              np.array([[1.0000000, -0.03793103, -0.3152709],
                                        [-0.03793103, 1.00000000, 0.0246941],
                                        [-0.3152709, 0.0246941, 1.00000000]]),
                              DP)


def test_2_vector_api(data: np.ndarray):
    assert np.isclose(corr(data[:, 0], data[:, 1])[0, 1], -0.1687177, atol=DP)

    data[-1, 0] = np.nan

    with np.errstate(invalid='ignore'):
        # Adding ignore because comparing nans raise invalid runtime warning
        assert np.isnan(corr(data[:, 0], data[:, 1])[0, 1])
    assert np.isclose(corr(data[:, 0], data[:, 1], use='complete')[0, 1], -0.16634284, atol=DP)
    assert np.isclose(corr(data[:, 0], data[:, 1], use='complete')[0, 1], -0.16634284, atol=DP)


def test_raise_error_if_invalid_method(data: np.ndarray):
    with pytest.raises(ValueError):
        corr(data, method='WRONG METHOD')


def test_raise_error_if_invalid_use(data: np.ndarray):
    with pytest.raises(ValueError):
        corr(data, use='SOMETHING STRANGE')


def test_raise_error_if_only_input_not_matrix():
    with pytest.raises(ValueError):
        corr(np.zeros(10))


def test_raise_error_if_x_y_different_length():
    with pytest.raises(ValueError):
        corr(np.zeros(10), np.ones(5))


def test_raise_error_if_x_y_not_vector():
    with pytest.raises(ValueError):
        corr(np.zeros((10, 2)), np.zeros(10))

    with pytest.raises(ValueError):
        corr(np.zeros(10), np.zeros((10, 2)))

    with pytest.raises(ValueError):
        corr(np.zeros((10, 2)), np.zeros((10, 2)))
