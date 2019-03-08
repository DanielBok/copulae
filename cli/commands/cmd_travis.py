import click

from cli.utils import *


@click.group()
def cli():
    """List of commands to setup package for travis"""
    pass


@cli.command()
def deps():
    """Builds dependency file for travis to install"""
    file = 'requirements.txt'
    echo("Building", style(file, 'green'))

    dependencies = shell_run('pip freeze').splitlines()

    ignore = ['conda', 'conda-build', 'conda-verify', 'menuinst', 'mkl-fft', 'mkl-random', 'pywin32', 'pywinpty']

    text = ""
    for d in dependencies:
        if d.startswith('-e'):
            # cli installation
            continue

        pkg = d.split('==')[0]
        if pkg in ignore:
            print(pkg)
            continue

        text += d + '\n'

    has_changed = write_file(ROOT.joinpath(file).as_posix(), text)

    if has_changed:
        n = len(text.strip().splitlines())
        echo(f"Wrote a total of {style(n, 'red')} dependencies into {style(file, 'green')}")
    else:
        echo(f"Contents were similar. {style(file, 'green')} not overwritten")

    return 0
