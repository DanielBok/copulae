Introduction to Copulae
-----------------------

A `copula <https://en.wikipedia.org/wiki/Copula_(probability_theory)>`_  is a multivariate probability distribution for which the marginal probability distribution of each variable  is uniform. It is a very popular form of dependency modelling between many disparate processes with different underlying distributions. 

Copula, Explained Simply
~~~~~~~~~~~~~~~~~~~~~~~~

Imagine you are in a factory which makes a `Nintendo Switch`__. To manufacture a Switch, you need parts coming from 3 different processes (they could even be manufactured in different countries). Let's call them :math:`A`, :math:`B`, :math:`C`.

If the failure rate of each process repectively are (for example) given by an `Exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_, a `Weibull <https://en.wikipedia.org/wiki/Weibull_distribution>`_ and a `Beta <https://en.wikipedia.org/wiki/Beta_distribution>`_ distribution individually, how do we model the chance of them failing together? (That would be catastrophe for all `Pokémons <https://en.wikipedia.org/wiki/Pok%C3%A9mon>`_!)

If only we could assume that all processes were independent of one another. Or if only we could assume the failure rates of all processes were given by a `Normal <https://en.wikipedia.org/wiki/Normal_distribution>`_ distribution, then we could dump a covariance matrix that explains the processes' relationships. 

But hope is not lost, (Pokémons rejoice!) this is where copulas (and Copulae) comes in.


Current State
~~~~~~~~~~~~~

Presently, only elliptical and `Archimedean copulas`__ have been implemented. I'm working on putting more common copulas into the package. I'll also be adding some common charts and statistical goodness-of-fit functions that can help the lay person more quickly learn the concepts. I struggled a bunch trying to pick this up, hope you'll have an easier time!

__ https://www.nintendo.com/switch/
__ https://en.wikipedia.org/wiki/Copula_(probability_theory)#Archimedean_copulas
