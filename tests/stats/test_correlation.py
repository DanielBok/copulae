import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal

from copulae.stats import corr, kendall_tau, pearson_rho, spearman_rho

DP = 3


@pytest.fixture
def data():
    np.random.seed(8888)
    return np.random.uniform(-10, 10, size=(30, 3))


@pytest.mark.parametrize('as_df', [True, False])
def test_corr(data, as_df):
    expected = np.array([[1.0000000, -0.16871767, -0.23407923],
                         [-0.1687177, 1.00000000, 0.05121912],
                         [-0.2340792, 0.05121912, 1.00000000]])

    if as_df:
        columns = [f'V{i + 1}' for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=columns)
        assert_frame_equal(corr(data), pd.DataFrame(expected, index=columns, columns=columns))
    else:
        assert_array_almost_equal(corr(data), expected, DP)


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


@pytest.mark.parametrize("as_series, x_name, y_name, exp_x_name, exp_y_name", [
    (True, 'X1', 'X2', 'X1', 'X2'),
    (True, '', '', 'X', 'Y'),
    (True, None, None, 'X', 'Y'),
    (True, 'V', 'V', 'V1', 'V2'),
    (False, '', '', '', '')])
@pytest.mark.parametrize("use", ['everything', 'complete', 'pairwise.complete'])
def test_2_vector_api(data: np.ndarray, as_series, use, x_name, y_name, exp_x_name, exp_y_name):
    expected = -0.1687177
    x, y = data[:, 0], data[:, 1]

    if as_series:
        x = pd.Series(x, name=x_name)
        y = pd.Series(x, name=y_name)

    res = corr(x, y, use=use)

    if as_series:
        assert np.isclose(res.to_numpy()[0, 1], expected, atol=DP)
        name = [exp_x_name, exp_y_name]
        assert_frame_equal(res, pd.DataFrame(res.to_numpy(), index=name, columns=name))
    else:
        assert np.isclose(res[0, 1], expected, atol=DP)


@pytest.mark.parametrize("use, expected", [
    ('everything', np.nan),
    ('complete', -0.16634284),
    ('pairwise.complete', -0.16634284),
])
def test_2_vector_with_nan(data, use, expected):
    data[-1, 0] = np.nan

    if np.isnan(expected):
        with np.errstate(invalid='ignore'):
            # Adding ignore because comparing nans raise invalid runtime warning
            assert np.isnan(corr(data[:, 0], data[:, 1], use=use)[0, 1])
    else:
        assert np.isclose(corr(data[:, 0], data[:, 1], use=use)[0, 1], expected, atol=DP)


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
