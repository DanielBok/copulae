import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from copulae.stats import multivariate_normal as mvn
from copulae.stats import multivariate_t as mvt

cov = np.array([
    [6.11380899, 2.78146876],
    [2.78146876, 1.2804278]
])

data = np.array([
    [2.20832204274957, 0.997979467727862],
    [-2.6883233124982, -1.35068190270785],
    [0.237049270165554, -0.0750615106959147],
    [1.21653537003823, 0.523740165669293],
    [3.4324360989112, 1.64078947261179],
    [5.72768640408248, 2.56918176517796],
    [0.595342483447132, 0.443991538933844],
    [5.84190727473856, 2.41160591899689],
    [3.56232518970767, 1.42826678407159],
    [-3.38325883488845, -1.48705661015132],
])


def test_mvt_pdf():
    expected = [0.303994245754277, 0.13658135276839, 0.13848015323344, 0.422736577196239, 0.131649388800426,
                0.0397860134559407, 0.151415052767095, 0.0133083319426572, 0.0539368361793873, 0.151613558641285]

    res = mvt.pdf(data, cov=cov, df=4)

    assert_array_almost_equal(res, expected)

    expected = [-1.19074650619494, -1.99083485091199, -1.97702826157638, -0.861006042869588, -2.02761303566035,
                -3.22423984915948, -1.88773051909607, -4.31936497804581, -2.919941617433, -1.88642037282532]
    log_res = mvt.logpdf(data, cov=cov, df=4)

    assert_array_almost_equal(log_res, expected)

    expected = [0.352148547892681, 0.169086597141526, 0.171537338150048, 0.452097714437596, 0.162674953282724,
                0.0343534018090551, 0.187971332076141, 0.00427939560451497, 0.0542022798009524, 0.188220017385623]

    norm_res = mvt.pdf(data, cov=cov, df=mvt._T_LIMIT + 1)
    assert_array_almost_equal(norm_res, expected)


@pytest.mark.parametrize('mean', [None, [4, 5]])
@pytest.mark.parametrize('df', [10, mvt._T_LIMIT + 1])
@pytest.mark.parametrize('size', [10, [10, 2]])
@pytest.mark.parametrize('type_', ['shifted', 'kshirsagar'])
def test_rvs_generates_correctly(mean, df, size, type_):
    rt = mvt.rvs(mean=mean, cov=cov, df=df, size=size, type_=type_, random_state=8)
    rn = mvn.rvs(cov=cov, size=size)

    assert rt.shape == rn.shape


@pytest.mark.parametrize('df, size, type_', [
    (None, 1, 'shifted'),
    (-1, 1, 'shifted'),
    (4, 4.5, 'shifted'),
    (4, [4, 4.5], 'shifted'),
    (4, 10, 'WRONG_TYPE'),
])
def test_rvs_raises_errors(df, size, type_):
    with pytest.raises(ValueError):
        mvt.rvs(cov=cov, df=df, size=size, type_=type_)


def test_process_parameters():
    # test scalar cov becomes covariance matrix
    assert mvt._process_parameters(None, None, 2, 4)[2].ndim == 2

    # test scalar mean gets propagated to vector mean with same length as covariance matrix
    assert len(mvt._process_parameters(None, 4.5, cov, 10)[1]) == len(cov)

    assert mvt._process_parameters(None, 5, cov, None)[3] == 4.6692


def test_wrong_parameters_raises_errors():
    with pytest.raises(ValueError):
        mvt._process_parameters([4.5], None, cov, 10)

    with pytest.raises(ValueError):
        mvt._process_parameters(None, [4.5] * 3, cov, 10)

    with pytest.raises(ValueError):
        mvt._process_parameters(None, None, cov - 14, 10)

    with pytest.raises(ValueError):
        mvt._process_parameters(None, None, cov, -10)
