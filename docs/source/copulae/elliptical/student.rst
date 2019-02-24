Student Copulas
===============

The implementation of the Student (T) copula is such that all the univariate marginal
distributions are student and the multivariate joint distribution is a multivariate-student
distribution.

Note that the degrees-of-freedom parameter is shared by all univariate margins as well as the
joint multivariate distribution

.. module:: copulae.elliptical.student
.. py:currentmodule:: copulae.elliptical

Student
~~~~~~~~
.. autoclass:: StudentCopula
   :inherited-members:
