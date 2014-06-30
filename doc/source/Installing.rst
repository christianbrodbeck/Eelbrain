.. highlight:: rst

Installing
==========

The easiest way to use the Eelbrain library is to install it from the 
`Python Package Index <https://pypi.python.org/pypi/eelbrain>`_ 
with ``easy_install``::

    $ easy_install eelbrain

And it can be updated similarly::

    $ easy_install -U eelbrain

.. note:: 
    Since version 0.3 Eelbrain comes with a C extension. For Intel Macs, 
    ``easy_install`` has access to a precompiled "egg" and should install 
    automatically as before. ``Pip`` on the other hand always compiles from 
    source and requires a recent version of XCode.


.. _dependencies:

Optional Dependencies
---------------------

The following modules provide additional functionality if they are installed:
    
* `rpy2 <http://rpy.sourceforge.net>`_ - in order to install it, first install 
  `R <http://www.r-project.org>`_, then use pip to install ``rpy2``: 
  ``$ pip install rpy2``.
* A working `LaTeX <http://www.latex-project.org/>`_ installation (enables 
  exporting tables as pdf).
* `wxPython <http://www.wxpython.org>`_ for using the GUI based on pyshell. 
  WxPython can be installed through the `EPD <https://www.enthought.com>`_. 
  `Currently it can not be installed through distutils 
  <http://stackoverflow.com/q/477573/166700>`_. 
  Installers are provided `here <http://www.wxpython.org/download.php>`_. 


.. _obtain-source:

Installing from GitHub
----------------------

The Eelbrain source code is hosted on `GitHub 
<https://github.com/christianbrodbeck/Eelbrain>`_. The source for the latest
development version can be downloaded as a 
`zip archive <https://github.com/christianbrodbeck/Eelbrain/zipball/master>`_.
However, since the code is evolving, the better option is to clone 
the project with git. A way to do this is::

    $ cd /target/directory
    $ git clone https://github.com/christianbrodbeck/Eelbrain.git

This will create the folder ``/target/directory/Eelbrain`` containing all the 
source files.

The source can then always be updated to the latest version
from within the ``Eelbrain`` directory::

    $ cd /target/directory/Eelbrain
    $ git pull

If Eelbrain is installed in ``develop`` mode, changes in the source folder 
(e.g., after running ``$ git pull``) take effect without re-installing::

	$ cd /target/directory/Eelbrain
	$ python setup.py develop

Besides installing the ``eelbrain`` module, this installs a shell script so 
that the pyshell terminal/editor optimized for OS X can be launched with::

    $ eelbrain 


.. _OS-X-app:

Creating Eelbrian.app on OS X
-----------------------------

On Mac OS X, a small .App file can be created which will launch the WxPython
GUI.

.. note::
    This step requires wxPython (see :ref:`optional dependencies 
    <dependencies>` above).

If using the enthought distribution, newer versions of some packages are 
required::

    $ enpkg --remove py2app macholib altgraph modulegraph
    $ easy_install pip
    $ pip install --upgrade py2app macholib altgraph modulegraph

Eelbrain should first be installed as package if this has not been done::

    $ cd /target/directory/Eelbrain
    $ python setup.py develop

The application can then be generated with::

    $ python setup.py py2app -A

This will create a small application in 
:file:`/target/directory/Eelbrain/dist/Eelbrain.app`. You can copy this application 
to your Applications folder (or anywhere else). However, the application file 
keeps references to the original source (due to the ``-A`` flag), 
so you must leave the source folder intact. 
The advantage of this method is that any 
changes in the source (such as ``$ git pull``) will be 
reflected as soon as you restart the application.
