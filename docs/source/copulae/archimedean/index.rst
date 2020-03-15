Archimedean Copulas
===================

Similar to all other copulas, Archimedean copulas are of the form

.. math::
   :nowrap:

    \begin{align*}
        H(\textbf{x}) = C(F_1(x_1), \dots, F_d(x_d)) \quad \textbf{x} \in \mathbb{R}^d
    \end{align*}

However, more specifically, an Archimedean is defined as

.. math::
   :nowrap:

    \begin{align*}
        C(\textbf{u}) = \psi(\psi^{-1}(u_1), \dots, \psi^{-1}(u_d)) \quad \textbf{u} \in [0, 1]^d
    \end{align*}

So in this class of copulas, you would first need a generator function, defined as :math:`\psi`. This function has the nice property that it is defined by a single value :math:`\theta`. Thus when you fit an Archimedean copula, you only need to "learn" this unknow value :math:`\theta`. 

In general, to be an Archimedean generator, :math:`\psi` must be a function that 

1. Is continuous and decreasing. This means that it maps a value :math:`x` from :math:`[0, \infty] \rightarrow [0, 1]`.
2. Has derivatives up to :math:`k = d-2` where :math:`d` is the dimension of the copula and that :math:`(-1)^k\psi^k(x) \geq 0 \quad \forall k \in \{0, \dots, d-2\}, x \in (0, \infty)`

One such function is the exponential function. In fact, if we let :math:`\psi(x) = \exp(-x), t \in [0, \infty]`, we would get the Gumbel copula.


.. toctree::
    :hidden:

    Clayton Copula <clayton>
    Frank Copula <frank>
    Gumbel Copula <gumbel>
