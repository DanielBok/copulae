Elliptical Copulas
==================

Elliptical copulas are very widely used copulas, perhaps the most. Essentially, the elliptical copula models where the univariate margins are joined by an elliptical distribution.

Similar to all other copulas, elliptical copulas are of the form

.. math::
   :nowrap:

    \begin{align*}
        H(\textbf{x}) = C(F_1(x_1), \dots, F_d(x_d)) \quad \textbf{x} \in \mathbb{R}^d
    \end{align*}

where, :math:`H` is the multivariate elliptical distribution function and :math:`F_1, \dots,F_d` 
correspond to the univariate margins.


.. toctree::
    :hidden:

    Gaussian Copula <gaussian>
    Student Copula <student>
