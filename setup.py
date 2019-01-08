#!/usr/bin/env python

from setuptools import find_packages, setup
import versioneer

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
]

setup(
    name='copulae',
    author='Daniel Bok',
    author_email='danielbok@gic.com.sg',
    maintainer='TG-EIS TPS Team',
    maintainer_email='grptgeis@gic.com.sg',
    packages=find_packages(include=['copulae', 'copulae.*']),
    license="MIT",
    version=versioneer.get_version(),
    description='Python copulas library for dependency modelling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://eisr',
    keywords=['copulae'],
    install_requires=requirements,
    setup_requires=setup_requirements,
    python_requires='>=3.5',
    platform='any',
    zip_safe=False,
)
