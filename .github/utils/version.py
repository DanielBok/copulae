import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path

root = Path(__file__).parents[2]

p = ArgumentParser(
    "Version updater",
    usage="python .github/utils/version.py [major|minor|patch|0.0.0]",
    description="""
Manage Copulae version

Run this and commit before creating a release in Github. When the release is created,
a new package is created and uploaded. Since the package takes its version from the
__version__ in copulae/__init__.py, it is important to update this variable before
creating the release. 
""".strip()
)
p.add_argument("version", type=str, help="New package version")

try:
    subprocess.run(["git", "--version"])
except FileNotFoundError as e:
    raise RuntimeError("git is not installed on your machine")

if __name__ == '__main__':
    # manage versions
    args = p.parse_args()
    version: str = args.version.lower().strip()

    file = root / "copulae" / "__init__.py"
    with open(file) as f:
        content = f.read()

    vs = re.findall(r'__version__ = "(\d+\.\d+\.\d+)"', content)
    if len(vs) == 0:
        raise RuntimeError("Cannot find version in __init__.py")

    current_version: str = vs.pop()
    if current_version == version:
        print("Versions are similar, nothing to change")

    if re.match(r"\d+\.\d+\.\d+", version):
        pass
    elif version in ('major', 'minor', 'patch'):
        major, minor, patch = [int(i) for i in current_version.split('.')]
        if version == 'major':
            major += 1
            minor, patch = 0, 0
        elif version == 'minor':
            minor += 1
            patch = 0
        elif version == 'patch':
            patch += 1
        version = f"{major}.{minor}.{patch}"
    else:
        raise RuntimeError(f"unrecognized version command: {version}. Use 'major', 'minor', 'patch' or specify "
                           f"a version number in the form 'a.b.c' where the letters are all numbers (semver)")

    content = re.sub(r'__version__ = "(\d+\.\d+\.\d+)"', f'__version__ = "{version}"', content)
    with open(file, 'w') as f:
        f.write(content)

    subprocess.run(["git", "commit", "-am", f"Bumped version to {version}"])
