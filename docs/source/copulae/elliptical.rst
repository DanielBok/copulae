Elliptical Copulas
==================

.. module::copulae.elliptical
.. py:currentmodule:: copulae.elliptical

Elliptical copulas are very widely used copulas, perhaps the most. Essentially, the elliptical copula models where the univariate margins are joined by an elliptical distribution.

Similar to all other copulas, elliptical copulas are of the form

.. math::
   :nowrap:

    \begin{align*}
        H(\textbf{x}) = C(F_1(x_1), \dots, F_d(x_d)) \quad \textbf{x} \in \mathbb{R}^d
    \end{align*}

where, :math:`H` is the multivariate elliptical distribution function and :math:`F_1, \dots,F_d` 
correspond to the univariate margins.

Gaussian
~~~~~~~~

The implementation of the Gaussian (Normal) copula is such that all the univariate marginal distributions are normal and the multivariate joint distribution is a multivariate-normal distribution.


.. autoclass:: GaussianCopula
   :inherited-members:


Student
~~~~~~~~

The implementation of the Student (T) copula is such that all the univariate marginal distributions are student and the multivariate joint distribution is a multivariate-student distribution.

Note that the degrees-of-freedom parameter is shared by all univariate margins as well as the joint multivariate distribution

.. autoclass:: StudentCopula
   :inherited-members:
