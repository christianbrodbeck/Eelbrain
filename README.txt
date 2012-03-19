Version: 0.0.4dev


About Eelbrain
==============

Eelbrain contains data analysis scripts and a Python shell bsed on wxPython.


Installing Eelbrain
===================

Eelbrain is a collection of pure Python scripts and does not require 
compilation. If all the dependencies are installed, it can be started
with::

    $ python eelbrain.py


Dependencies
------------

Eelbrain requires Python 2.6.x or 2.7.x.  Depending on which components are
used there are different dependencies. Several additional modules add optional
functions.

* For running data processing and statistics

	* numpy
	* scipy
	* matplotlib

* for the wxterm interface (optional)

	* wxPython

* for optional features:
	
	* tex (plus working latex distribution) for exporting tables as pdf
	* bioread


Creating an OS X Application
============================

An OS X application can be created using py2app. An alias application 
can be created using::

	$ python setup.py py2app -A

This creates a slim Eelbrain.app file in ``dist/Eelbrain.app`` which 
references the source code in the directory of the setup.py file. 
``Eelbrain.app`` can be copied anywhere on the computer, but the folder 
containing the ``setup.py`` file and its subfolders have to be left in place
because ``Eelbrain.app`` uses them. In addition, any changes to the source code
will be reflected in ``Eelbrain.app``.

