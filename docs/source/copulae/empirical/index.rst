Empirical Copula
================

.. py:currentmodule:: copulae.empirical

The empirical distribution function is a natural nonparametric estimator of a distribution function. Given a copula :math:`C`, a non-parametric estimator of :math:`C` is given by

.. math::

    C_n(\mathbf{u}) &= \frac{1}{n} \sum^n_{i=1} 1(\mathbf{U}_{i, n} \leq \mathbf{u} \\
        &= \frac{1}{n} \sum^n_{i=1} \prod^d_{j=1} 1(U_{ij, n} \leq u_j)  \quad \mathbf{u} \in [0, 1]^d

The estimator above is the default used by the :class:`EmpiricalCopula` class when smoothing is set to :code:`None`.

The empirical copula, being a particular multivariate empirical distribution function, often exhibits a large bias when the sample size is small. One way to counteract this is to use the empirical beta copula. The estimator is given by

.. math::

    C^\beta_n(\mathbf{u}) = \frac{1}{n} \sum^n_{i=1} \prod^d_{j=1} F_{n, R_{ij}}(u_j)  \quad \mathbf{u} \in [0, 1]^d

where :math:`F_{n, r}` represents a beta distribution function with parameters :math:`r` and :math:`n + 1 - r` and where :math:`R_{ij}` represents the rank of :math:`X_{ij}` where :math:`\mathbf{X}` is the original data set used to "fit" the empirical copula.

Another smooth version of the empirical copula estimator is the "checkerboard" copula. Its estimator is given by

.. math::

    C^\#_n(\mathbf{u}) = \frac{1}{n} \sum^n_{i=1} \prod^d_{j=1} \min\{\max\{nu_j - R_{i,j} + 1, 0\}, 1\}  \quad \mathbf{u} \in [0, 1]^d


.. autoclass:: EmpiricalCopula
    :members:
    :inherited-members:
