import sys
from pathlib import Path

import click

cmd_folder: Path = Path(__file__).absolute().parent.joinpath('commands')
cmd_prefix = 'cmd_'


class CopulaeCLI(click.MultiCommand):

    def get_command(self, ctx: click.Context, name: str):
        filepath = cmd_folder.joinpath(f'{cmd_prefix}{name}.py').as_posix()

        ns = {}
        try:
            with open(filepath) as f:
                code = compile(f.read(), filepath, 'exec')
                eval(code, ns, ns)

            return ns['cli']
        except FileNotFoundError:
            click.secho(f"Command {name} does not exists. ", file=sys.stderr, fg='red')

    def list_commands(self, ctx: click.Context):
        _commands = sorted(cmd_folder.glob(f'{cmd_prefix}*.py'))
        return [f.name[len(cmd_prefix):-3] for f in _commands]


@click.command(cls=CopulaeCLI)
def main():
    """
    Commands to manage the project
    """
    pass
