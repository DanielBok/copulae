import pytest

from copulae.stats.uniform import random_uniform


@pytest.mark.parametrize('n', [1, 2, 4])
@pytest.mark.parametrize('dim', [1, 3, 5])
def test_random_uniform(n, dim):
    assert random_uniform(n, dim, 8).shape == (n, dim)
