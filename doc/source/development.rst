***********
Development
***********

The latest development version can be found in the master branch in the
`GitHub repository <https://github.com/christianbrodbeck/Eelbrain>`_.

New contributions can be made as pull requests into that branch.

Style guides:

- Python: `PEP8 <https://www.python.org/dev/peps/pep-0008>`_
- Documentation: `NumPy documentation style
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
- `ReStructured Text Primer <http://sphinx-doc.org/rest.html>`_


Useful tools:

 - Graphical frontend for git: `GitX <http://rowanj.github.io/gitx/>`_
 - Python IDE: `PyCharm <https://www.jetbrains.com/pycharm/>`_


Set up the Development Version
------------------------------

Set up your own fork of `Eelbrain
<https://github.com/christianbrodbeck/Eelbrain>`_
(see `GitHub's instructions for working with a fork
<https://help.github.com/articles/fork-a-repo/>`_).

Make sure Xcode is installed (open it once to accept the license agreement).
Some command line tools provided by Xcode are needed to compile the Cython
extension included in Eelbrain.

In order to start working with the development version of Eelbrain, install it
with the ``develop`` option (from within the project directory)::

    $ python setup.py develop

In Python, you can make sure that you are working with the development version::

    >>> import eelbrain
    >>> eelbrain.__version__
    'dev'

To switch back to the release version use ``$ pip uninstall eelbrain``.


Testing
-------

Eelbrain uses `nose <https://nose.readthedocs.org>`_ for testing. To run all
tests, run ``$ make test`` from the Eelbrain project directory. Tests for
individual modules are included in folders called "tests", usually on the same
level as the module.
Preferably, new features should be added with tests.
