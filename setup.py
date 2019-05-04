#!/usr/bin/env python
import sys

from setuptools import Extension, find_packages, setup

import versioneer

ENV = 'DEV'
argv = sys.argv
for e in argv:
    if e.startswith('--env'):
        _, ENV = e.upper().split('=')
        argv.remove(e)

IS_DEV_MODE = ENV == 'DEV'

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

AUTHOR = 'Daniel Bok'
EMAIL = 'daniel.bok@outlook.com'

with open('README.md') as f:
    long_description = f.read()

requirements = [
    'numba >=0.43',
    'numpy >=1.15',
    'scipy >=1.1',
    'pandas >=0.23',
    'statsmodels >=0.9'
]

extras_require = {
    'dev': [
        'pytest',
        'pytest-cov',
        'twine'
    ]
}


def build_ext_modules():
    macros = [('NPY_NO_DEPRECATED_API', '1'),
              ('NPY_1_7_API_VERSION', '1')]

    modules = [
        {
            'name': 'copulae.gof._exchtest',
            'sources': ['copulae/gof/_exchtest'],
        },
    ]

    extensions = []
    for m in modules:
        # if not built with Cython, use the c or cpp files
        language = m.get('language', 'c')
        ext = '.pyx' if USE_CYTHON else f'.{language}'

        for i, source in enumerate(m['sources']):
            m['sources'][i] = source + ext

        extensions.append(Extension(**m, language=language, define_macros=macros))

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


def run_setup():
    setup(
        name='copulae',
        author=AUTHOR,
        author_email=EMAIL,
        maintainer=AUTHOR,
        maintainer_email=EMAIL,
        packages=find_packages(include=['copulae', 'copulae.*']),
        license="MIT",
        version=versioneer.get_version(),
        description='Python copulae library for dependency modelling',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/DanielBok/copulae',
        keywords=['copula', 'copulae', 'dependency modelling', 'dependence structures', 'archimdean', 'elliptical',
                  'finance'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Financial and Insurance Industry',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
        ],
        install_requires=requirements,
        extras_require=extras_require,
        include_package_data=True,
        python_requires='>=3.6',
        zip_safe=False,
        ext_modules=build_ext_modules(),
    )


if __name__ == '__main__':
    run_setup()
