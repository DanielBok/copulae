import os
import shutil

import click

from cli.utils import ROOT, echo, style


@click.command()
def cli():
    """Removes unnecessary artifacts after build and tests"""
    deleted_items = clean()
    if deleted_items is None:
        return 1

    echo('Deleted', style(len(deleted_items), 'red'), 'items')
    for i, (typ, name) in enumerate(deleted_items):
        echo('\t',
             style(f'{i + 1:2d}', 'green'),
             f': {name} ',
             style(f'[{typ}]', 'yellow' if typ == 'Folder' else 'blue'),
             sep='')
    return 0


def clean():
    echo("Cleaning up")

    # files and folders to destroy
    to_delete = [
        '.coverage',
        'build',
        'copulae.egg-info',
        'copulae_cli.egg-info',
        'dist',
        'docs/build',
        '.eggs',
        '.pytest_cache'
    ]

    try:
        deleted_items = []
        for item in to_delete:
            path = ROOT.joinpath(item)
            if path.exists():
                i = path.as_posix()
                if path.is_file():
                    os.remove(i)
                    deleted_items.append(('File', i))
                else:
                    shutil.rmtree(i)
                    deleted_items.append(('Folder', i))

        return deleted_items
    except FileNotFoundError as e:
        echo(str(e), err=True)
        return None
