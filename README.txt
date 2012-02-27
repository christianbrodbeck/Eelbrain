About Eelbrain
==============

Eelbrain contains data analysis scripts and a Python shell bsed on ``wx.py``.


Installing Eelbrain
===================

Eelbrain is a collection of pure Python scripts and does not require 
compilation. 


Dependencies
------------

Eelbrain requires Python 2.6.x or 2.7.x.  Depending on which components are
used there are different dependencies. Several additional modules add optional
functions.

* For running data processing and statistics

	* numpy
	* scipy
	* matplotlib

* For the wxterm interface 

	* wxPython

* optional:
	
	* tex (plus working latex distribution) for exporting tables as pdf
	* bioread

* for OS X Integration (create .app file)

	* setuptools
	* py2app

* for Windows Integration (create .exe.file)

	* py2exe


Installing
==========

OS X
----

Create Eelbrain.app using::

	$ python setup.py py2app -A

This creates a slim Eelbrain.app file in ``dist/Eelbrain.app`` which 
references the source code in the directory of the setup.py file. 
``Eelbrain.app`` can be copied anywhere on the computer, but the folder 
containing the ``setup.py`` file and its subfolders have to be left in place
because ``Eelbrain.app`` uses them. In addition, any changes to the source code
will be reflected in ``Eelbrain.app``.


Windows
-------

Blahblah
 