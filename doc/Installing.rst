.. highlight:: rst

Installing
==========

Eelbrain is a pure Python project and uses 
`Distribute <http://packages.python.org/distribute/setuptools.html>`_, 
so required dependencies are automatically installed from the Python Package
Index. In order to use Eelbrain, 

#.  Depending on your purpose, install :ref:`optional dependencies 
    <dependencies>`
#.  :ref:`Obtain the Eelbrian source code <obtain-source>`
#.  Install and run Eelbrain in one of those ways:

    a. :ref:`Install as a package <install-package>`
    b. On OS X, :ref:`create an Eelbrain.app Application <OS-X-app>`
       with :py:mod:`py2app`


.. _dependencies:

Optional Dependencies
---------------------

The following modules provide additional functionality if they are installed:
    
* `wxPython <http://www.wxpython.org>`_ 
  for using the GUI based on pyshell.
  It `seems <http://stackoverflow.com/q/477573/166700>`_ that currently 
  wxPython can not be installed through distutils. 
  Installers are provided
  `here <http://www.wxpython.org/download.php>`_. 
  For installing on EPD-64 bit on OS X, see :ref:`below <EPD64>`.
* `mne <https://github.com/mne-tools/mne-python>`_
* `tex <http://pypi.python.org/pypi/tex>`_ Enables exporting tables as pdf 
  (also requires a working `LaTeX <http://www.latex-project.org/>`_ installation)
* `bioread <http://pypi.python.org/pypi/bioread>`_ Enables an importer for 
  ``.acq`` files.


.. _EPD64:

EPD-64 bit on OS X
------------------

EPD-64 bit comes without wxPython, but the latest development version of
wxPython can be installed manually.
First, make sure the right Python distribution (and *only* the right one) is 
added to the ``PATH`` in ``~/.bash_profile``. 

Install wxPython from source::

    $ wget http://downloads.sourceforge.net/wxpython/wxPython-src-2.9.4.0.tar.bz2
    $ open wxPython-src-2.9.4.0.tar.bz2 
    $ cd wxPython-src-2.9.4.0/wxPython
    $ sudo python build-wxpython.py --build_dir=../bld --osx_cocoa --install

EPD-64 7.3 seems to come with a deficient version of `MDP 
<http://mdp-toolkit.sourceforge.net>`_. If MDP is required, it can be replaced 
with `pip <http://www.pip-installer.org/>`_. 
Install pip (unless it is already installed)::

    $ curl https://raw.github.com/pypa/pip/master/contrib/get-pip.py | sudo python

Then, install MDP from github::

    $ sudo pip install -e git://github.com/mdp-toolkit/mdp-toolkit#egg=MDP


.. _obtain-source:

Installing from GitHub
----------------------

The Eelbrain source code is hosted on `GitHub 
<https://github.com/christianmbrodbeck/Eelbrain>`_. The latest source can be 
downloaded as a 
`zip archive <https://github.com/christianmbrodbeck/Eelbrain/zipball/master>`_.
However, since the code is currently evolving, the better option is to clone 
the project with git. A way to do this is::

    $ cd /target/directory
    $ git clone https://github.com/christianmbrodbeck/Eelbrain.git

This will create the folder ``/target/directory/Eelbrain`` containing all the 
source files.


Updating
^^^^^^^^

The source can then always be updated to the latest version
from within the ``Eelbrain`` directory::

    $ cd /target/directory/Eelbrain
    $ git pull

After obtaining the source, there are several options to use Eelbrain:

.. note::
    Make sure to run setup.py with the python version with which you want to
    use Eelbrain.



.. _install-package:

A. Install as a Package
-----------------------

If you install Eelbrain as a package, you can use it in two ways:

- import as a module
- launch as an application

I recommend to install Eelbrain in ``develop`` mode. This has the
benefit that changes in the source folder (e.g., after running 
``$ git pull``) take effect without re-installing::

	$ cd /target/directory/Eelbrain
	$ python setup.py develop

Besides installing the ``eelbrain`` module, this installs a shell script so 
that Eelbrain can be launched with::

    $ eelbrain 


.. _OS-X-app:

B. Create Eelbrian.app on OS X
------------------------------

.. note::
    Invoking ``$ python setup.py py2app`` does not seem to properly
    take care of dependencies. For this reason, Eelbrain should
    be :ref:`installed as package <install-package>` before invoking the 
    ``py2app`` build command.

The application can be generated with::

    $ cd /target/directory/Eelbrain
    $ python setup.py py2app -A

This will create a small application in 
:file:`/target/directory/Eelbrain/dist/Eelbrain.app`. You can copy this application 
to your Applications folder (or anywhere else). However, the application file 
keeps references to the original source (due to the ``-A`` flag), 
so you must leave the source folder intact. 
The advantage of this method is that any 
changes in the source (such as ``$ git pull``) will be 
reflected as soon as you restart the application.

