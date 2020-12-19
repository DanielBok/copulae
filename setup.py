#!/usr/bin/env python
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, find_packages, setup

IS_DEV_MODE = False
argv = sys.argv
for e in argv:
    if e.startswith('--dev'):
        IS_DEV_MODE = True

try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options
    from Cython.Distutils.build_ext import new_build_ext as build_ext

    USE_CYTHON = True
    Options.annotate = IS_DEV_MODE
except ImportError:
    from distutils.command.build_ext import build_ext

    USE_CYTHON = False


    def cythonize(x, *args, **kwargs):
        return x


    class Options:
        pass

requirements = [
    'numpy >=1.19',
    'pandas >=1.1',
    'scikit-learn >=0.23',
    'scipy >=1.5',
    'statsmodels >=0.12.1',
    "typing-extensions >=3.7.4.3; python_version < '3.8'",
    "wheel >=0.36",
    'wrapt >=1.12'
]

setup_requires = [
    'cython',
    'numpy',
    'scipy'
]


def build_ext_modules():
    macros = [('NPY_NO_DEPRECATED_API', '1'),
              ('NPY_1_7_API_VERSION', '1')]

    if IS_DEV_MODE:
        macros.append(('CYTHON_TRACE', '1'))

    if platform.system() == 'Windows':
        parallelism_options = {'extra_compile_args': ['/openmp']}
    elif platform.system() == 'Linux':
        parallelism_options = {'extra_compile_args': ['-fopenmp'],
                               'extra_link_args': ['-fopenmp']}
    else:  # Darwin, MACOS
        parallelism_options = {}

    extensions = []
    for root, _, files in os.walk("copulae"):
        path_parts = os.path.normcase(root).split(os.sep)
        for file in files:
            fn, ext = os.path.splitext(file)

            if ext == '.pyx':
                module_path = '.'.join([*path_parts, fn])
                _fp = os.path.join(*path_parts, fn)
                pyx_c_file_path = _fp + ('.pyx' if USE_CYTHON else '.c')

                include_dirs = []
                with open(_fp + ext) as f:
                    if re.search(r'^cimport numpy as c?np$', f.read(), re.MULTILINE) is not None:
                        include_dirs.append(np.get_include())

                extensions.append(Extension(
                    module_path,
                    [pyx_c_file_path],
                    language='c',
                    include_dirs=include_dirs,
                    define_macros=macros,
                    **parallelism_options
                ))

    # compiler directives
    compiler_directives = {
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'cdivision': True,
        'language_level': '3',
        'linetrace': IS_DEV_MODE,
        'profile': IS_DEV_MODE,
    }

    return cythonize(extensions, compiler_directives=compiler_directives)


def get_git_version():
    file = Path(__file__).parent / "copulae" / "__init__.py"
    with open(file, 'r') as f:
        matches = re.findall(r'__version__ = "(\S+)"', f.read())

    if len(matches) != 1:
        raise RuntimeError("could not find package version")

    return matches[0]


setup(
    packages=find_packages(include=['copulae', 'copulae.*']),
    version=get_git_version(),  # get latest tagged version
    setup_requires=setup_requires,
    install_requires=requirements,
    zip_safe=False,
    ext_modules=build_ext_modules(),
)
