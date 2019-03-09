import click

from cli.utils import *


@click.group()
def cli():
    """List of commands to setup docs"""
    pass


@cli.command()
def build():
    """Builds the docs"""
    shell_run('make html', cwd=ROOT.joinpath('docs'))


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
