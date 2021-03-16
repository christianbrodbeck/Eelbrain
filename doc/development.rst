***********
Development
***********

Eelbrain is hosted on `GitHub <https://github.com/christianbrodbeck/Eelbrain>`_.


The Development Version
-----------------------

Development takes place on the ``master`` branch, while release versions are maintained on
branches called ``r/0.26`` etc. For further information on working with
GitHub see `GitHub's instructions <https://help.github.com/articles/fork-a-repo/>`_.

The repository contains a conda environment that contains everything needed to use Eelbrain except Eelbrain itself.
To generate the ``eeldev`` environment, use::

    $ conda env create --file=env-dev.yml

The development version of Eelbrain can then be installed through ``setup.py``::

    $ conda activate eeldev
    $ python setup.py develop

On macOS, the ``$ eelbrain`` shell script to run ``iPython`` with the framework
build is not installed properly by ``setup.py``; in order to fix this, run::

    $ ./fix-bin


In Python, you can make sure that you are working with the development version::

    >>> import eelbrain
    >>> eelbrain.__version__
    'dev'


Contributing
------------

Contributions to code and documenation are welcome as pull requests into the ``master`` branch.

Style guides:

- Python code style follows `PEP8 <https://www.python.org/dev/peps/pep-0008>`_ (mostly)
- The documentation is written in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext>`_
- Docstrings follow the `numpydoc style  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_


Useful tools:

- Graphical frontend for git: `SourceTree <https://www.sourcetreeapp.com>`_
- Python IDE: `PyCharm <https://www.jetbrains.com/pycharm>`_
- Faster ``conda`` alternative: `Mamba <https://github.com/mamba-org/mamba>`_


Testing
-------

Tests for individual modules are included in folders called ``tests``, usually
on the same level as the module.
To run all tests, run ``$ make test`` from the Eelbrain project directory.
On macOS, tests needs to run with the framework build of Python;
if you get a corresponding error, run ``$ ./fix-bin pytest`` from the
``Eelbrain`` repository root.
