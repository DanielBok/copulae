import numpy as np

__all__ = ['random_uniform']


def random_uniform(n: int, dim: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(size=(n, dim))
