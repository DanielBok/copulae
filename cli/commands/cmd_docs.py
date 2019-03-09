import shutil

import click

from cli.utils import *


@click.group()
def cli():
    """List of commands to setup docs"""
    pass


@cli.command()
@click.option('--clean', 'clean_', is_flag=True, default=False, help='If enabled, removes all previous builds')
def build(clean_):
    """Builds the docs"""
    if clean_:
        status = _remove_build_dir()
        if status != 0:
            return status

    echo("Building docs.. ", nl=False)
    shell_run('make html', cwd=ROOT.joinpath('docs'))
    echo(style('Done', 'green'))

    return 0


@cli.command()
def clean():
    """Removes previous docs build"""
    return _remove_build_dir()


@cli.command()
def deps():
    """Builds dependency file for read the docs to install"""
    file = 'requirements_docs.txt'
    echo("Building", style(file, 'green'))

    dependencies = shell_run('pip freeze').splitlines()

    capture = ['sphinx']

    text = ""
    for d in dependencies:
        if d.startswith('-e'):
            # cli installation
            continue

        pkg = d.split('==')[0].lower()
        for c in capture:
            if c in pkg.lower():
                text += d + '\n'

    has_changed = write_file(ROOT.joinpath(file).as_posix(), text)

    if has_changed:
        n = len(text.strip().splitlines())
        echo(f"Wrote a total of {style(n, 'red')} dependencies into {style(file, 'green')}")
    else:
        echo(f"Contents were similar. {style(file, 'green')} not overwritten")

    return 0


def _remove_build_dir():
    folder1 = ROOT.joinpath('docs', 'build')
    folder2 = ROOT.joinpath('docs', 'source', '_build')
    try:
        if any([folder1.exists(), folder2.exists()]):
            echo('Removing build folder in docs.. ', nl=False)
            for f in (folder1, folder2):
                if f.exists():
                    shutil.rmtree(f)

            echo(style('Successful', 'green'))

        return 0
    except FileNotFoundError as e:
        echo(str(e), err=True)
        return 1
