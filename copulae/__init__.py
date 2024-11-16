from copulae.archimedean import *
from copulae.core import pseudo_obs
from copulae.elliptical import *
from copulae.empirical import *
from copulae.indep import *
from copulae.marginal import *
from copulae.mixtures import *


def doc():
    import webbrowser
    webbrowser.open('https://copulae.readthedocs.io/en/latest')


from copulae._version import get_versions

v = get_versions()
__version__ = v.get("closest-tag", v["version"])
__git_version__ = v.get("full-revisionid")
del get_versions, v
