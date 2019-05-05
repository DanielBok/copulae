from copulae._version import get_versions
from copulae.archimedean import *
from copulae.elliptical import *
from copulae.indep import *

__version__ = get_versions()['version']
del get_versions


def doc():
    import webbrowser
    webbrowser.open('https://copulae.readthedocs.io/en/latest')
