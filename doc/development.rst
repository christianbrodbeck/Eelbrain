***********
Development
***********

Eelbrain is actively developed and maintained by
`Christian Brodbeck <http://loop.frontiersin.org/people/120185>`_
at the `Computational sensorimotor systems lab
<http://www.isr.umd.edu/Labs/CSSL/simonlab/Home.html>`_
at University of Maryland, College Park.


.. _obtain-source:

The Development Version
-----------------------

The Eelbrain source code is hosted on
`GitHub <https://github.com/christianbrodbeck/Eelbrain>`_.
To clone the repository, use::

    $ git clone https://github.com/christianbrodbeck/Eelbrain.git


Installing as Development Version
---------------------------------

Installation requires the presence of a compiler.
On macOS, make sure Xcode is installed (open it once to accept the license
agreement).
Windows will indicate any needed files when the install command is run.

After cloning the repository, the development version can be installed by
running, from the ``Eelbrain`` root directory::

    $ python setup.py develop


On macOS, the ``$ eelbrain`` shell script to run iPython with the framework
build is not installed properly by ``setup.py``; in order to fix this, run

    $ ./fix-bin

For further information on working with GitHub see `GitHub's instructions
<https://help.github.com/articles/fork-a-repo/>`_.

In Python, you can make sure that you are working with the development version::

    >>> import eelbrain
    >>> eelbrain.__version__
    'dev'

To switch back to the release version use ``$ pip uninstall eelbrain``.


Building with Conda
-------------------

To Build Eelbrain with ``conda``, make sure that you are in your root
environment and that ``conda-build`` is installed. Then, from
``Eelbrain/conda`` run::

    $ conda build eelbrain --numpy=1.11

This builds against numpy 1.11, to build against another version replace
``1.11``.
After building successfully, the build can be installed with::

    $ conda install --use-local eelbrain


Contributing
------------

Eelbrain is fully open-source and you are welcome to contribute.
New contributions can be made as pull requests into the master branch on GitHub.

I am focusing development on macOS, in particular for the GUI components.
However, the underlying libraries are all available on the major platforms and
it would be possible to make Eelbrain work on other platform with little effort.


Style guides:

- Python: `PEP8 <https://www.python.org/dev/peps/pep-0008>`_
- Documentation: `NumPy documentation style
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
- `ReStructured Text Primer <http://sphinx-doc.org/rest.html>`_


Useful tools:

 - Graphical frontends for git:
   - `GitX <http://rowanj.github.io/gitx>`_
   - `SourceTree <https://www.sourcetreeapp.com>`_
 - Python IDE: `PyCharm <https://www.jetbrains.com/pycharm>`_


Testing
-------

Eelbrain uses `nose <https://nose.readthedocs.org>`_ for testing. To run all
tests, run ``$ make test`` from the Eelbrain project directory. Tests for
individual modules are included in folders called "tests", usually on the same
level as the module.
