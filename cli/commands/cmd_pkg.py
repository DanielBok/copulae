import shutil
from pathlib import Path

import click

from cli.utils import *


@click.group()
def cli():
    """Tools to build and manage Copulae packaging"""
    pass


@cli.command()
@click.option('--clean', 'clean_', is_flag=True, default=False, help='If enabled, removes all previous builds')
def build(clean_):
    """Builds sdist and bdist wheel"""
    if clean_:
        status = _remove_build_dir()
        if status != 0:
            return status

    return _build_package()


@cli.command()
def clean():
    """Removes all build artifacts"""
    return _remove_build_dir()


@cli.command()
@click.option('-u', '--user', 'user', default=None,
              help='PYPI username. Defaults to value in .pypirc if available')
@click.option('-p', '--password', 'password', default=None,
              help='PYPI password. Defaults to value in .pypirc if available')
@click.option('-b', '--build', 'build_', is_flag=True, default=False, help='If enabled, rebuilds package')
@click.option('--clean', 'clean_', is_flag=True, default=False, help='If enabled, removes all previous builds')
@click.option('--verbose', 'verbose', is_flag=True, default=True, help='Verbose output')
def upload(user, password, build_, clean_, verbose):
    """Uploads built package to PYPI"""
    cmd = [f"twine upload"]

    if clean_:
        status = _remove_build_dir()
        if status != 0:
            return status

        _build_package()  # cleaning automatically rebuilds package. If not there won't be any package to upload

    elif build_:
        _build_package()

    pypirc = Path.home().joinpath('.pypirc').read_text().splitlines()
    if user is None:
        for line in pypirc:
            if line.lower().startswith('username'):
                user = line.split()[-1].strip()
                break
        else:
            user = click.prompt("Enter your username", default='DannieBee', type=str)

    if password is None:
        for line in pypirc:
            if line.lower().startswith('password'):
                password = line.split()[-1].strip()
                break
        else:
            password = click.prompt("Enter your password", hide_input=True, type=str)

    cmd.append(f'--user {user} --password {password}')
    if verbose:
        cmd.append('--verbose')

    cmd.append('dist/*')

    echo(f'Uploading packages to PyPI')
    shell_run(*cmd, tty=True)

    return 0


def _build_package():
    shell_run("make dist", tty=True)
    return 0


def _remove_build_dir():
    folder1 = ROOT.joinpath('build')
    folder2 = ROOT.joinpath('dist')
    try:
        if any([folder1.exists(), folder2.exists()]):
            echo('Removing previous package build folder artifacts.. ', nl=False)
            for f in (folder1, folder2):
                if f.exists():
                    shutil.rmtree(f)

            echo(style('Successful', 'green'))

        return 0
    except FileNotFoundError as e:
        echo(str(e), err=True)
        return 1
