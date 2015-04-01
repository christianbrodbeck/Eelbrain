**********
Installing
**********

.. note::
    Eelbrain comes with a C extension. Precompiled binaries are currently
    provided for OS X. On other platforms, installing Eelbrain requires
    compilation.


.. contents:: Contents
   :local:


Installing from PYPI
--------------------

.. note::
   If you are setting up a new Python environment, the easier options might be
   `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ or
   `Canopy <https://www.enthought.com/products/canopy>`_.

Eelbrain can be installed from the
`Python Package Index (PYPI) <https://pypi.python.org/pypi/eelbrain>`_::

    $ pip install eelbrain

This will not install those dependencies that require compilation, since it
might be easier to install them from a different source. Eelbrain can be
installed with all its dependencies from the PYPI with::

    $ pip install eelbrain[full]

And, if you want to use pysurfer::

    $ pip install eelbrain[plot.brain]

Later, update Eelbrain with::

    $ pip install -U eelbrain


`WxPython <http://www.wxpython.org>`_ is not strictily necessary but enhances
plots with several GUI elements (and enables the epoch rejection GUI).
wxPython `can not be installed from the PYPI
<http://stackoverflow.com/q/477573/166700>`_, but installers are provided
`here <http://www.wxpython.org/download.php>`__.


Installing with Anaconda
------------------------

For using Eelbrain with Continuum Analytics' `Anaconda
<https://store.continuum.io/cshop/anaconda/>`_, first install the dependencies
with ``conda``::

    $ conda install wxpython
    $ conda install mayavi

Then install Eelbrain::

    $ easy_install eelbrain

For plotting MNE source estimates, also install PySurfer::

    $ pip install pysurfer

Later, update Eelbrain with::

    $ pip uninstall eelbrain
    $ easy_install -U eelbrain



Installing with Canopy
----------------------

Make sure that you are using the
`Canopy <https://www.enthought.com/products/canopy>`_ Python distribution from
the command line (see
`here <https://support.enthought.com/entries/23646538-Make-Canopy-User-Python-be-your-default-Python-i-e-on-the-PATH->`__).

Install dependencies that do not come with Canopy by default. The only
additional package needed is Mayavi::

   $ enpkg mayavi

To make sure you are working with the latest versions of all dependencies,
update the Canopy distribution before installing or updating Eelbrain::

   $ enpkg --update-all

Install Eelbrain::

   $ pip install eelbrain

For plotting MNE source estimates, also install PySurfer::

    $ pip install pysurfer

Later, update Eelbrain with::

   $ pip install -U eelbrain


Optional Dependencies
---------------------

The following items provide additional functionality if they are installed:

* `rpy2 <http://rpy.sourceforge.net>`_ - in order to install it, first install
  `R <http://www.r-project.org>`_, then use pip to install ``rpy2``:
  ``$ pip install rpy2``.
* A working `LaTeX <http://www.latex-project.org/>`_ installation (enables
  exporting tables as pdf).


.. _obtain-source:

Installing the development version from GitHub
----------------------------------------------

The Eelbrain source code is hosted on
`GitHub <https://github.com/christianbrodbeck/Eelbrain>`_. The source for the
latest development version can be downloaded as a
`zip archive <https://github.com/christianbrodbeck/Eelbrain/zipball/master>`_.
However, since the code is evolving, the better option is to clone the project
with git. A way to do this is::

    $ cd /target/directory
    $ git clone https://github.com/christianbrodbeck/Eelbrain.git

This will create the folder ``/target/directory/Eelbrain`` containing all the
source files.

The source can then always be updated to the latest version from within the
``Eelbrain`` directory::

    $ cd /target/directory/Eelbrain
    $ git pull

If Eelbrain is installed in ``develop`` mode, changes in the source folder
(e.g., after running ``$ git pull``) take effect without re-installing::

	$ cd /target/directory/Eelbrain
	$ python setup.py develop
