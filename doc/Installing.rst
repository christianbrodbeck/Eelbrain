.. highlight:: rst

Installing
==========

Eelbrain is a pure Python project and uses 
`Distribute <http://packages.python.org/distribute/setuptools.html>`_, 
so most dependencies are automatically installed. In order to use Eelbrain, 

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
    
* `wxPython <http://www.wxpython.org>`_ for using the GUI based on pyshell. 
  WxPython can be installed through the `EPD <https://www.enthought.com>`_. 
  `Currently it can not be installed through distutils 
  <http://stackoverflow.com/q/477573/166700>`_. 
  Installers are provided `here <http://www.wxpython.org/download.php>`_. 
* `mne <https://github.com/mne-tools/mne-python>`_
* A working `LaTeX <http://www.latex-project.org/>`_ installation (enables 
  exporting tables as pdf).


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
    This step requires wxPython (see :ref:`optional dependencies 
    <dependencies>` above).

Invoking ``$ python setup.py py2app`` does not seem to properly
take care of dependencies. For this reason, Eelbrain should
be :ref:`installed as package <install-package>` before invoking the 
``py2app`` build command.

The application can then be generated with::

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
