import json
import subprocess
import sys
from os import path, getcwd


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


if __name__ == '__main__':
    convert()
