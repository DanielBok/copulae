import os
import shutil

root = os.path.dirname(os.path.dirname(__file__))


def clean():
    folders = ['build', 'dist', 'copulae.egg-info']
    for folder in folders:
        shutil.rmtree(folder)


if __name__ == '__main__':
    clean()
