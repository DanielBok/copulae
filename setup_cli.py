# run this with
# python setup_cli.py develop
from setuptools import setup, find_packages

import versioneer

AUTHOR = 'Daniel Bok'
EMAIL = 'daniel.bok@outlook.com'

setup(
    name='copulae-cli',
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    version=versioneer.get_version(),
    install_requires=[
        'Click'
    ],
    include_package_data=True,
    packages=find_packages(include=['cli', 'cli.*']),
    entry_points="""
        [console_scripts]
        copulae=cli.main:main
    """
)
