"""
setup.py
========

This is the setup.py script for Eelbrain.

http://docs.python.org/distutils/index.html
https://setuptools.readthedocs.io/en/latest/setuptools.html


About MANIFEST.in:

https://docs.python.org/2/distutils/sourcedist.html#manifest-template

"""
# Setuptools bootstrap module
# http://pythonhosted.org//setuptools/setuptools.html
from ez_setup import use_setuptools
use_setuptools('17')

from packaging.version import Version
import os
import re
from setuptools import setup, find_packages, Extension

import numpy as np

# Distributing Cython modules
# https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html#distributing-cython-modules
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False


DESC = """
GitHub: https://github.com/christianbrodbeck/Eelbrain
"""
IS_WINDOWS = os.name == 'nt'

# version must be in X.X.X format, e.g., "0.0.3dev"
with open('eelbrain/__init__.py') as fid:
    text = fid.read()
match = re.search(r"__version__ = '([.\w]+)'", text)
if match is None:
    raise ValueError("No valid version string found in:\n\n" + text)
version = match.group(1)
Version(version)  # check that it's a valid version

# Cython extensions
args = {'define_macros': [("NPY_NO_DEPRECATED_API", "NPY_1_11_API_VERSION")]}
if IS_WINDOWS:
    open_mp_args = {**args, 'extra_compile_args': '/openmp'}
else:
    open_mp_args = {
        **args,
        'extra_compile_args': ['-Wno-unreachable-code', '-fopenmp', '-O3', '-mavx'],
        'extra_link_args': ['-fopenmp'],
    }
    args['extra_compile_args'] = ['-Wno-unreachable-code', '-O3', '-mavx']
ext = '.pyx' if cythonize else '.c'
ext_cpp = '.pyx' if cythonize else '.cpp'
extensions = [
    Extension('eelbrain._data_opt', [f'eelbrain/_data_opt{ext}'], **args),
    Extension('eelbrain._trf._boosting_opt', [f'eelbrain/_trf/_boosting_opt{ext}'], **open_mp_args),
    Extension('eelbrain._ndvar._convolve', [f'eelbrain/_ndvar/_convolve{ext}'], **open_mp_args),
    Extension('eelbrain._stats.connectivity_opt', [f'eelbrain/_stats/connectivity_opt{ext}'], **args),
    Extension('eelbrain._stats.opt', [f'eelbrain/_stats/opt{ext}'], **args),
    Extension('eelbrain._stats.vector', [f'eelbrain/_stats/vector{ext_cpp}'], include_dirs=['dsyevh3C'], **args),
]
if cythonize:
    extensions = cythonize(extensions)

setup(
    name='eelbrain',
    version=version,
    description="MEG/EEG analysis tools",
    url="http://eelbrain.readthedocs.io",
    author="Christian Brodbeck",
    author_email='christianbrodbeck@nyu.edu',
    license='BSD (3-clause)',
    long_description=DESC,
    python_requires='>=3.8',
    setup_requires=[
        "numpy >= 1.20",
        "cython >= 3",
    ],
    extras_require={
        'base': [
            'colormath >= 2.1',
            'keyring >= 5',
            'mne >= 0.19',
            'nibabel >= 2.5',
            'pillow',
            'tqdm >= 4.40',
        ],
        'full': [
            'numpy >= 1.20',
            'scipy >= 1.3',
            'matplotlib >= 3.3.4',
        ],
        'plot.brain': [
            'pysurfer[save_movie] >= 0.9',
        ],
    },
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=extensions,
    scripts=['bin/eelbrain'],
)
