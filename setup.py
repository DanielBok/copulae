#!/usr/bin/env python
from setuptools import find_packages, setup

import versioneer

with open('README.md') as f:
    long_description = f.read()

requirements = [
    'numpy >=1.15',
    'scipy >=1.1',
    'pandas >=0.23',
    'statsmodels >=0.9'
]

setup_requirements = [
    'pytest',
    'pandas',
]


# def build_extensions():
#     directives = {'language_level': '3'}
#     Options.annotate = True
#
#     return cythonize(extensions, compiler_directives=directives)


def run_setup():
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
        setup_requires=setup_requirements,
        include_package_data=True,
        python_requires='>=3.5',
        zip_safe=False,
    )


if __name__ == '__main__':
    run_setup()
