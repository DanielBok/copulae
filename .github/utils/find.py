import os
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional


def _create_parser():
    p = ArgumentParser(
        "Find and copy files",
        usage="python .github/utils/find.py copy -from <from> -to <to>",
        description="""
Used to replace the linux find function because we need some magic that the 'find' function
doesn't provide. 

Currently, only the 'copy' and 'remove' function are supported. During copy, 
the basename of the file is kept (all the parent paths are stripped).

When running remove, the current working directory is searched recursively for all the
files matching the pattern.
""".strip()
    )

    p.add_argument("action",
                   type=str,
                   help="The action to execute, currently only `copy` or `remove`".strip())

    p.add_argument("-from",
                   type=str,
                   help="""
Directory to copy from. If it does not start with '/', the folder is assumed to be 
relative from the directory command was run from (relative path)""".strip(),
                   default="")

    p.add_argument('-to',
                   type=str,
                   help="""
Directory to copy to. If it does not start with '/', the folder is assumed to be 
relative from the directory command was run from (relative path). Note that even
we never change the filename.""".strip(),
                   default="")

    p.add_argument('-pattern',
                   type=str,
                   help="Regex file pattern to match")

    p.add_argument("-dry_run", action="store_true")

    return p


def find_and_copy(from_folder: Path,
                  to_folder: Path,
                  pattern: Optional[str],
                  dry_run: bool):
    if pattern is None:
        pat = None
    else:
        pat = re.compile(pattern)

    print(f"Copying files from {from_folder} to {to_folder}")

    for file in Path(from_folder).rglob('*'):
        if file.is_file():
            if pat is None or pat.search(file.as_posix()):
                print(f"Copying file '{file}' to {to_folder / file.name}")
                if not dry_run:
                    shutil.copy(file, to_folder / file.name)


def find_and_remove(pattern: str, dry_run: bool):
    for file in Path.cwd().rglob(pattern):  # type: Path
        if file.is_file():
            print(f"Removing file: '{file}'")
            if not dry_run:
                os.remove(file)


def create_path(path: str):
    fp = Path(path)
    if not path.startswith('/'):
        fp = Path.cwd() / fp
    return fp


if __name__ == '__main__':
    parser = _create_parser()
    args = parser.parse_args()

    action = args.action.lower().strip()
    if action == "copy":
        find_and_copy(create_path(getattr(args, 'from')),
                      create_path(getattr(args, 'to')),
                      args.pattern,
                      args.dry_run)
    elif action == "remove":
        find_and_remove(args.pattern, args.dry_run)
