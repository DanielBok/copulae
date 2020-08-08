import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path

root = Path(__file__).parent

p = ArgumentParser(
    "Version updater",
    usage="python version.py [major|minor|patch|0.0.0]",
    description="Manage Copulae version"
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

    # commit to git
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z (windows) to save it.")
    message = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        message.append(line)

    message = '\n'.join(message).strip() + '\n'

    subprocess.run(["git", "commit", "-am", "Bumped version number"])
    subprocess.run(["git", "tag", "-a", version, "-m", message], shell=True)
