#!/usr/bin/env python
from setuptools import find_packages, setup

import versioneer

AUTHOR = 'Daniel Bok'
EMAIL = 'daniel.bok@outlook.com'

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
    'pytest-cov',
    'pandas',
]


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
            'License :: OSI Approved :: MIT',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
        ],
        install_requires=requirements,
        setup_requires=setup_requirements,
        include_package_data=True,
        python_requires='>=3.6',
        zip_safe=False,
    )


if __name__ == '__main__':
    run_setup()
