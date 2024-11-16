"""Custom build extensions for Cython compilation"""
import platform

import numpy
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        if self.include_dirs is None:
            self.include_dirs = []
        self.include_dirs.append(numpy.get_include())

    def build_extensions(self):
        # Read dev mode settings from pyproject.toml
        for ext in self.extensions:
            ext.include_dirs.append(numpy.get_include())

            ext.define_macros.extend([
                ("NPY_NO_DEPRECATED_API", 1),
                ("NPY_1_7_API_VERSION", 1)
            ])

            if platform.system() == 'Windows':
                ext.extra_compile_args.append('/openmp')
            elif platform.system() == 'Linux':
                ext.extra_compile_args.append('-fopenmp')
                ext.extra_link_args.append('-fopenmp')

        # Cythonize with settings from pyproject.toml
        self.extensions = cythonize(
            self.extensions,
            compiler_directives={
                'language_level': 3,
                'binding': True,
                'wraparound': False,
                'boundscheck': False,
                'nonecheck': False,
                'cdivision': True,
            },
            include_dirs=[numpy.get_include()],
        )
        super().build_extensions()
