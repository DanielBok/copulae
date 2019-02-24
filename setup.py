#!/usr/bin/env python
import os
from setuptools import Extension, find_packages, setup
import versioneer

from Cython.Build import cythonize
from Cython.Compiler import Options

with open('README.md') as f:
    long_description = f.read()

requirements = [
    'numpy >=1.15.0',
    'scipy >=1.1.0',
    'statsmodels >=0.9.0'
]

setup_requirements = [
    'pytest',
    'pandas',
    'cython >=0.29',
]


def build_extensions():
    extensions = [
        Extension('copulae.stats.cdf.beta',
                  ['copulae/stats/cdf/beta.pyx'])
    ]

    # for path, _, files in os.walk('copulae'):
    #     for file in files:
    #         file_path, ext = os.path.splitext(file)
    #         if ext.endswith('.pyx'):
    #             sources = [os.path.join(path, file)]
    #             e = Extension()

    directives = {'language_level': '3'}
    Options.annotate = True

    return cythonize(extensions, compiler_directives=directives)


# def _build_ext(folder: str, file: str):
#     file = file.casefold()
#     file_path, ext = os.path.splitext(file)
#
#     if ext != '.pyx':
#         return None
#
#     folder = folder.casefold()
#     file_path = os.path.join(folder, file)
#
#     with open(file_path) as f:
#         contents = f.read()
#
#     include_numpy = False
#     for line in contents.split('\n'):
#         line = line.strip()
#         if line.startswith('cimport numpy'):
#             include_numpy = True
#
#         if line.startswith('cdef extern from'):
#
#
#     return Extension


def run_setup():
    ext_modules = build_extensions()

    setup(
        name='copulae',
        author='Daniel Bok',
        author_email='danielbok@gic.com.sg',
        maintainer='TG-EIS TPS Team',
        maintainer_email='grptgeis@gic.com.sg',
        packages=find_packages(include=['copulae', 'copulae.*']),
        license="MIT",
        version=versioneer.get_version(),
        description='Python copulae library for dependency modelling',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://eisr',
        keywords=['copulae'],
        install_requires=requirements,
        ext_modules=ext_modules,
        setup_requires=setup_requirements,
        python_requires='>=3.5',
        zip_safe=False,
    )


if __name__ == '__main__':
    run_setup()
