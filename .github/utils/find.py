import re
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional


def _create_parser():
    p = ArgumentParser(
        "Find and copy files",
        usage="python find.py -from <from> -to <to>",
        description="""
Used to replace the linux find function because we need some magic that the 'find' function
doesn't provide. 

Currently, only the 'copy' function is supported. During copy, the basename of the file is
kept (all the parent paths are stripped).
""".strip()
    )

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

    return p


def find_and_copy(from_folder: Path,
                  to_folder: Path,
                  pattern: Optional[str]):
    if pattern is None:
        pat = None
    else:
        pat = re.compile(pattern)

    print(f"Copying files from {from_folder} to {to_folder}")

    for file in Path(from_folder).rglob('*'):
        if file.is_file():
            if pat is None or pat.search(file.as_posix()):
                print(f"Copying file '{file}' to {to_folder / file.name} ")
                shutil.copy(file, to_folder / file.name)


def create_path(path: str):
    fp = Path(path)
    if not path.startswith('/'):
        fp = Path.cwd() / fp
    return fp


if __name__ == '__main__':
    parser = _create_parser()
    args = parser.parse_args()

    find_and_copy(create_path(getattr(args, 'from')),
                  create_path(getattr(args, 'to')),
                  args.pattern)
