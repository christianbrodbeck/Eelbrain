Conda
=====

Configuring:

$ conda config --append channels conda-forge christianbrodbeck

check with

$ conda config --get channels


Adding a package
----------------

To add a package that is already on PYPI:

$ conda skeleton pypi <package>

$ conda build <package>

--python all

convert to other platform:

$ conda convert -p all /anaconda/conda-bld/osx-64/<package>.tar.bz2 -o /anaconda/conda-bld
$ anaconda upload /anaconda/conda-bld/win-64/<package>.tar.bz2


Eelbrain
--------

- Update version in ``setup.py`` and ``meta.yaml``
- run with appropriate numpy version:

    $ conda build eelbrain --numpy 1.11

- Upload as beta:

    $ anaconda upload -l beta /anaconda/conda-bld/<platform>/<package>.tar.bz2
