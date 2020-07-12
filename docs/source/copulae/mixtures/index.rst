Mixture Copula
==============

Mixture copulas are instances where

#. Multiple copulas are combined to model the data
    * As an example, imagine you have an Empirical Copula and Gaussian Copula meshed together to fit the data
#. A copula where the dependence structure is a combination of various distributions
    * An example would be the Gaussian Mixture Copula (GMC). In contrast to the Gaussian Copula where the dependence
      structure is a uni-modal distribution, the GMC's dependence structure is a mixture of many Gaussian distributions
      and is thus multi-modal.

Presently, only Gaussian Mixture Copula is implemented.

.. toctree::
    :caption: Mixture Copulas

    Gaussian Mixture Copula <gmc>
