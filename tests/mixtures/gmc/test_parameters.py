import pytest
from numpy.testing import assert_allclose

from copulae.core.exceptions import NonSymmetricMatrixError
from copulae.mixtures.gmc.exception import GMCParamError
from copulae.mixtures.gmc.parameter import GMCParam

param2 = GMCParam(
    n_clusters=3,
    n_dim=2,
    prob=[0.48923563, 0.05350188, 0.45726249],
    means=[
        [-7.866836, -11.073276],
        [-3.336661, 24.139628],
        [-23.879568, -5.213695]
    ],
    covs=[
        [[15.9705696, -0.27545926],
         [-0.2754593, 0.01738696]],
        [[3.973527, -2.588777],
         [-2.588777, 6.676961]],
        [[5.3549889, -0.3326589],
         [-0.3326589, 4.8624179]]
    ]
)

param3 = GMCParam(
    n_clusters=3,
    n_dim=3,
    prob=[0.48923563, 0.05350188, 0.45726248],
    means=[
        [-7.866836, -11.073276, -3.336661],
        [24.139628, -23.879568, -5.213695],
        [0.01579013, 19.96915555, -0.20905425],
    ],
    covs=[
        [[10.2117067, 1.2660601, 0.4483301],
         [1.2660601, 16.0272934, -0.2477051],
         [0.4483301, -0.2477051, 10.6160624]],

        [[10.577139, -2.0196453, 2.3501990],
         [-2.019645, 5.7465517, 0.8115075],
         [2.350199, 0.8115075, 6.0729275],
         ],
        [[7.21748434, 0.59955180, -0.06163302],
         [0.59955180, 7.14172519, -0.06657406],
         [-0.06163302, -0.06657406, 7.01162171]]
    ]
)

vector2 = [-0.04306413, -2.87305219, -0.17136819, -7.866836, -11.073276,
           -3.336661, 24.139628, -23.879568, -5.213695, 15.9705696,
           -0.27545926, 0.01738696, 3.973527, -2.588777, 6.676961,
           5.3549889, -0.3326589, 4.8624179]

vector3 = [-0.04306411, -2.87305218, -0.17136822, -7.866836, -11.073276,
           -3.336661, 24.139628, -23.879568, -5.213695, 0.01579013,
           19.96915555, -0.20905425, 10.2117067, 1.2660601, 0.4483301,
           16.0272934, -0.2477051, 10.6160624, 10.577139, -2.0196453,
           2.350199, 5.7465517, 0.8115075, 6.0729275, 7.21748434,
           0.5995518, -0.06163302, 7.14172519, -0.06657406, 7.01162171]


@pytest.mark.parametrize("param, expected", [
    (param2, vector2),
    (param3, vector3),
])
def test_to_vector(param, expected):
    vector = param.to_vector()
    assert_allclose(vector, expected, rtol=1e-4)


@pytest.mark.parametrize('vector, param', [
    (vector2, param2),
    (vector3, param3)
])
def test_from_vector(vector, param):
    actual = GMCParam.from_vector(vector, param.n_clusters, param.n_dim)
    assert_allclose(actual.prob, param.prob, rtol=1e-4)
    assert_allclose(actual.means, param.means, rtol=1e-4)
    assert_allclose(actual.covs, param.covs, rtol=1e-4)


@pytest.mark.parametrize("param", [param2, param3])
def test_from_dict(param):
    actual = GMCParam.from_dict({
        "prob": param.prob,
        "means": param.means,
        "covs": param.covs,
    })

    assert_allclose(actual.prob, param.prob, rtol=1e-4)
    assert_allclose(actual.means, param.means, rtol=1e-4)
    assert_allclose(actual.covs, param.covs, rtol=1e-4)


@pytest.mark.parametrize("key, value, error", [
    ("prob", [0.3, 0.3], GMCParamError),  # not sum to 1
    ("prob", [0.3, 0.3, 0.4], GMCParamError),  # do not match number of clusters
    ("means", [[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]], GMCParamError),  # bad shape
    ("means", [[0.2, 0.3], [0.2, 0.3], [0.2, 0.3]], GMCParamError),  # bad shape
    ("covs", [
        [[5, 6],
         [6, 9]],
        [[2, 3],
         [3, 5]],
        [[2, 3],
         [3, 5]]
    ], GMCParamError),  # bad shape
    ("covs", [
        [[27, 16, 35],
         [16, 30, 44],
         [35, 44, 77]],
        [[27, 16, 35],
         [16, 30, 44],
         [35, 44, 77]]
    ], GMCParamError),  # bad shape
    ("covs", [
        [[5, 3],
         [4, 0]],
        [[5, 3],
         [4, 0]]
    ], NonSymmetricMatrixError)  # not psd
])
def test_param_raises_error(key, value, error):
    payload = {
        # default values
        **{
            'prob': [0.5, 0.5],
            'means': [
                [0.1, 0.2],
                [0.3, 0.4]
            ],
            'covs': [
                [[5, 6],
                 [6, 9]],
                [[2, 3],
                 [3, 5]]
            ]
        },
        key: value  # additional value
    }
    with pytest.raises(error):
        GMCParam(2, 2, **payload)
