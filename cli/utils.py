import pathlib
import subprocess
from typing import Union

import click

__all__ = ['ROOT', 'echo', 'shell_run', 'style', 'write_file']

ROOT: pathlib.Path = pathlib.Path(__file__).cwd()


def echo(*msg: str, sep=' ', file=None, nl=True, err=False, color=None):
    """
    Prints a message plus a newline to the given file or stdout.

    :param msg: the message components to print
    :param sep: the character to join each message components
    :param file: the file to write to (defaults to ``stdout``)
    :param err: if set to true the file defaults to ``stderr`` instead of
                ``stdout``.  This is faster and easier than calling
                :func:`get_text_stderr` yourself.
    :param nl: if set to `True` (the default) a newline is printed afterwards.
    :param color: controls if the terminal supports ANSI colors or not.  The
                  default is autodetection.
    """
    message = sep.join(str(i) for i in msg)
    click.echo(message, file, nl, err, color)


def shell_run(*args, cwd: Union[str, pathlib.Path] = None, shell=True) -> str:
    if cwd is None:
        cwd = ROOT.as_posix()
    elif isinstance(cwd, pathlib.Path):
        cwd = cwd.as_posix()
    return subprocess.check_output(' '.join(str(a) for a in args), cwd=cwd, shell=shell).decode('utf-8')


def style(text, fg=None, bg=None, bold=None, dim=None, underline=None,
          blink=None, reverse=None, reset=True):
    """Styles a text with ANSI styles and returns the new string.  By
    default the styling is self contained which means that at the end
    of the string a reset code is issued.  This can be prevented by
    passing ``reset=False``.

    Examples::

        click.echo(click.style('Hello World!', fg='green'))
        click.echo(click.style('ATTENTION!', blink=True))
        click.echo(click.style('Some things', reverse=True, fg='cyan'))

    Supported color names:

    * ``black`` (might be a gray)
    * ``red``
    * ``green``
    * ``yellow`` (might be an orange)
    * ``blue``
    * ``magenta``
    * ``cyan``
    * ``white`` (might be light gray)
    * ``bright_black``
    * ``bright_red``
    * ``bright_green``
    * ``bright_yellow``
    * ``bright_blue``
    * ``bright_magenta``
    * ``bright_cyan``
    * ``bright_white``
    * ``reset`` (reset the color code only)

    :param text: the string to style with ansi codes.
    :param fg: if provided this will become the foreground color.
    :param bg: if provided this will become the background color.
    :param bold: if provided this will enable or disable bold mode.
    :param dim: if provided this will enable or disable dim mode.  This is
                badly supported.
    :param underline: if provided this will enable or disable underline.
    :param blink: if provided this will enable or disable blinking.
    :param reverse: if provided this will enable or disable inverse
                    rendering (foreground becomes background and the
                    other way round).
    :param reset: by default a reset-all code is added at the end of the
                  string which means that styles do not carry over.  This
                  can be disabled to compose styles.
    """

    return click.style(str(text), fg, bg, bold, dim, underline, blink, reverse, reset)


def write_file(file: Union[str, pathlib.Path], content: str):
    if isinstance(file, pathlib.Path):
        file = file.as_posix()

    try:
        with open(file, 'r') as f:
            original = f.read()

        is_diff = original.strip() != content.strip()
    except FileNotFoundError:
        is_diff = True

    if is_diff:
        with open(file, 'w') as f:
            f.write(content)

    return is_diff
