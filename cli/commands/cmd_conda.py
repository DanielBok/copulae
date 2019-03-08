import json
import subprocess
import sys
from os import path, getcwd

import click

from cli.utils import *


@click.group()
def cli():
    """Commands to manage conda environment setup and build processes"""
    pass


@cli.command()
def deps():
    """Builds the env.yaml file used for other devs to build their conda environment"""

    file = 'env.yaml'
    conda_env = shell_run('conda env export --name copulae').splitlines()

    echo(f"Exporting conda environment into {style(file, 'green')}")

    n_channel, n_deps = 0, 0
    texts = []
    state = ""

    for line in conda_env:
        if line.startswith('name'):
            texts.append(line)
            continue
        elif line.startswith('channels'):
            state = 'channels'
            texts.append(line)
            continue
        elif line.startswith('dependencies'):
            state = 'deps'
            texts.append(line)
            continue
        elif line.startswith('prefix'):
            continue

        if state == 'channels':
            channel = line.strip().split().pop()
            if channel in ('defaults', 'conda-forge', 'anaconda'):
                texts.append(line)
                n_channel += 1
        elif state == 'deps':
            texts.append(line)
            n_deps += 1

    has_changed = write_file(ROOT.joinpath(file), '\n'.join(texts))

    if has_changed:
        echo(f"Wrote a total of {style(n_deps, 'red')} channels and  {style(n_deps, 'red')} dependencies "
             f"into {style(file, 'green')}")
    else:
        echo(f"Contents were similar. {style(file, 'green')} not overwritten")

    return 0


def convert():
    """
    Converts the win-64 or (platform) packages for other platforms
    """

    args = sys.argv
    if len(args) == 1:
        print('No dist path specified, defaulting to root folder')
        dist = getcwd()
    else:
        dist = path.join(getcwd(), args[1])
        print(f'Dist folder: {dist}')

    # reading channel data for package details
    with open(path.join(dist, 'channeldata.json')) as f:
        data: dict = json.load(f)

    # getting package data
    keys = list(data['packages'].keys())
    if len(keys) != 1:
        raise KeyError("Must only have 1 package")

    name: str = keys.pop()
    details: dict = data['packages'][name]

    package_path = details["reference_package"]
    platform, package = path.split(package_path)

    # converting packages
    platforms = ['win-32', 'win-64', 'linux-32', 'linux-64']
    package_path = path.join(dist, package_path)
    for p in platforms:
        if platform.lower() == p:
            continue

        cmd = ["conda", "convert", "-p", p]
        if dist != '':
            cmd += ['-o', dist]
        cmd.append(package_path)

        subprocess.Popen(cmd).wait()
