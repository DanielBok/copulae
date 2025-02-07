from pathlib import Path

def cleanup_requirements():
    root = Path(__file__).parents[2]

    req_folder = root / "requirements"

    for file in req_folder.glob("*.txt"):  # type: Path
        with open(file) as f:
            content = f.read().strip()

        new_lines = []
        for line in content.splitlines():
            dep, *_ = line.split(";")
            if dep.startswith("typing-extensions"):
                dep = dep + '; python_version < "3.11"'
            new_lines.append(dep.strip())
        new_lines.append('')

        with open(file, 'w') as f:
            f.write("\n".join(new_lines))


if __name__ == '__main__':
    cleanup_requirements()
