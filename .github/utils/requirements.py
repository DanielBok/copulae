import re
import tomllib
from pathlib import Path


def cleanup_requirements():
    root = Path(__file__).parents[2]
    with open(root / "pyproject.toml", "rb") as f:
        py = tomllib.load(f)

    req_folder = root / "requirements"
    with open(req_folder / "test.txt", "w") as f:
        f.write(_form_dependencies(
            py['project']['dependencies'],
            py['tool']['poetry']['group']['test']['dependencies'],
            py['tool']['poetry']['group']['build']['dependencies'],
        ))

    with open(req_folder / "build.txt", "w") as f:
        f.write(_form_dependencies(
            py['tool']['poetry']['group']['build']['dependencies'],
        ))

def _form_dependencies(*sections):
    deps = {}
    for s in sections:
        new_deps = _parse_dependencies(s)
        deps = new_deps | deps

    items = list(deps.values())
    items.append("")
    return "\n".join(items)


def _parse_dependencies(dependencies: list[str] | dict[str, str]):
    output = {}
    if isinstance(dependencies, list):
        pattern = r"([\w-]+)\s*\(([\d\s,<>=.]+)\)(?:\s*;\s*(python_version\s*(?:<|>|<=|>=|==)\s*\"\d[(?:.\d)]+\"))?"
        pattern = re.compile(pattern)

        for line in dependencies:
            pkg, version, extras = pattern.search(line).groups()
            dep = f"{pkg}{version}"
            if extras:
                dep = f"{dep} ; {extras}"
            output[pkg] = dep

    if isinstance(dependencies, dict):
        for pkg, version in dependencies.items():  # type: str, str
            if version.startswith("^"):
                version = version[1:]
                next_major = int(version.split(".")[0]) + 1
                version = f">={version},<{next_major}"
            else:
                raise NotImplementedError
            output[pkg] = f"{pkg}{version}"

    return output



if __name__ == '__main__':
    cleanup_requirements()
