import os
import shutil

import click

from cli.utils import ROOT, echo, style


@click.command()
def cli():
    """Removes unnecessary artifacts after build and tests"""

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

        echo('Deleted', style(len(deleted_items), 'red'), 'items')
        for i, (typ, name) in enumerate(deleted_items):
            echo('\t',
                 style(f'{i + 1:2d}', 'green'),
                 f': {name} ',
                 style(f'[{typ}]', 'yellow' if typ == 'Folder' else 'blue'),
                 sep='')

        return 0
    except FileNotFoundError as e:
        echo(str(e), err=True)
        return 1
