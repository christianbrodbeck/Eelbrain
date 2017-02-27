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

from distutils.version import StrictVersion
import re
from setuptools import setup, find_packages

from Cython.Build import cythonize
import numpy as np


DESC = """
GitHub: https://github.com/christianbrodbeck/Eelbrain
"""

# version must be in X.X.X format, e.g., "0.0.3dev"
with open('eelbrain/__init__.py') as fid:
    text = fid.read()
match = re.search("__version__ = '([.\w]+)'", text)
if match is None:
    raise ValueError("No valid version string found in:\n\n" + text)
version = match.group(1)
if version != 'dev':
    s = StrictVersion(version)  # check that it's a valid version

# basic setup arguments
setup(
    name='eelbrain',
    version=version,
    description="MEG/EEG analysis tools",
    url="http://eelbrain.readthedocs.io",
    author="Christian Brodbeck",
    author_email='christianbrodbeck@nyu.edu',
    license='BSD (3-clause)',
    long_description=DESC,
    python_requires='==2.7',
    setup_requires=open('requirements_install.txt').read().splitlines(),
    tests_require=[
        'nose',
    ],
    install_requires=[
        'colormath >= 2.1',
        'keyring >= 5',
        'mne >= 0.13.1',
        'nibabel >= 2.0',
        'pillow',
        'tex >= 1.8',
        'tqdm >= 4.8',
    ],
    extras_require={
        'full': ['numpy >= 1.8',
                 'scipy >= 0.17',
                 'matplotlib >= 1.1'],
        'dev': ['sphinx >= 1.1',
                'numpydoc >= 0.5'],
        'plot.brain': ['pysurfer[save_movie] >= 0.7']
    },
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=cythonize('eelbrain/_stats/*.pyx'),
    scripts=['bin/eelbrain'],
)
