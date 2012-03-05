Eelbrain Interface
==================

WxPython Gui
------------

The Eelbrain GUI is based on `wxPython <http://www.wxpython.org/>`_. 


Getting Help
^^^^^^^^^^^^

* Pressing the ``f1`` key will open the help viewer. If the terminal or an 
  editor is currently active, the object help viewer will show the 
  docomentation for the object currently under the cursor. Alternatvely, you 
  can call ``>>> help(object)``.
* Many GUI elements (e.g., toolbar buttons) have a short-help which can be
  displayed by hovering the mouse pointer over it.


Debugging
^^^^^^^^^

If something crashes you can start the debugger with::

	>>> import pdb
	>>> pdb.pm()
	
To end the debugger use ``q`` for quit::

	(Pdb) q

For more information on pdb see the `pdb documentation <http://docs.python.org/library/pdb.html>`_.



Keyboard Shortcuts
------------------

Some keyboard shortcuts are available in all windows, others are available 
depending on the type of the active window. On Os-X, for many system commands
such as saving and opening files, replace ``ctrl`` with ``command``


General
^^^^^^^

* ``F1``: Open help for the command under the caret; in the help viewer the whole 
	object name must be selected.
* ``ctrl`` - ``w``: close top-most editor or help window 


Shell
^^^^^

(Most are standard PyShell shortcuts)

* Exporting text

	* ``ctrl`` - ``d``:  Copy the selected commands to the topmost editor window
	* ``Ctrl`` - ``C``:  Copy selected text, removing prompts.
	* ``Ctrl`` - ``Shift`` - ``C``:  Copy selected text, retaining prompts.
	* ``Alt`` - ``C``:  Copy to the clipboard, including prefixed prompts.
	* ``Ctrl`` - ``X``:  Cut selected text.
	* ``Ctrl`` - ``V``:  Paste from clipboard.
	* ``Ctrl`` - ``Shift`` - ``V``:  Paste and run multiple commands from clipboard.

* History

	* ``Ctrl`` - ``Up Arrow``:  Retrieve previous history item.
	* ``Alt`` - ``P``:  Retrieve previous history item.
	* ``Ctrl`` - ``Down Arrow``:  Retrieve next history item.
	* ``Alt`` - ``N``:  Retrieve next history item.
	* ``Shift`` - ``Up Arrow``:  Insert previous history item.
	* ``Shift`` - ``Down Arrow``:  Insert next history item.
	* ``F8``:  Command-completion of history item. (Type a few characters of a previous 
	  command and press F8.)

* Moving the caret/selection

	* ``Home``:  Go to the beginning of the command or line.
	* ``Shift`` - ``Home``:  Select to the beginning of the command or line.
	* ``Shift`` - ``End``:  Select to the end of the line.
	* ``End``:  Go to the end of the line.
	* ``Ctrl`` - ``F``:  Search 
	* ``F3``:  Search next
	* ``Ctrl`` - ``H``:  "Hide" lines containing selection / "unhide"

* Editing

	* ``Ctrl`` - ``Enter``: Insert new line into multiline command.
	* ``F12``: on/off "free-edit" mode

* Font

	* ``Ctrl`` - ``]``: Increase font size.
	* ``Ctrl`` - ``[``: Decrease font size.
	* ``Ctrl`` - ``=``: Default font size.

* Auto completion

	* ``Ctrl``-``Space``: Show Auto Completion.
	* ``Ctrl``-``Alt``-``Space``: Show Call Tip.
	* ``Shift`` - ``Enter``: Complete Text from History.


Editor
^^^^^^

* ``ctrl`` - ``/``:  Comment or uncomment selected lines
* ``ctrl`` - ``s``:  Save current document
* ``alt`` - ``arrow (up/down)``:  Move current line up or down (!!! uses copy-paste)


Shell Commands
--------------

The following commands are available in the shell in addition to normal Python
commands. For more information, use help(command):

.. py:function:: attach(object)

	brings properties of object into the global namespace (for example variables 
	from an Experiment object)

.. py:function:: clear()

	clear the shell

.. py:function:: copy(object)

	copy str(object) to the clipboard

.. py:function:: help([object])

	retrieve help for any object 

.. py:function:: loadtable([filename])

	load a table from a file

.. py:function:: printdict(dictionary)

	prints a more readable representation for complex dictionaries


.. py:function:: table([list])

	open a simple table editor. Can create a table from a 2 dimensional list as argument


Modules
-------

The following Python modules are imported by default:

External Modules
^^^^^^^^^^^^^^^^

*	``np``: `NumPy <http://numpy.scipy.org/>`_ (foundation for numerical computing in Python)
*	``sp``: `SciPy <http://www.scipy.org/>`_ (builds on numpy, providing many scientific functions)
*	``P``: `matplotlib <http://matplotlib.sourceforge.net/>`_.pylab  (quick plotting with matplotlib)
*	``mpl``: `Matplotlib <http://matplotlib.sourceforge.net/>`_  (object-oriented plotting)
*	``wx``: `wxPython <http://www.wxpython.org/>`_  (user interface components)


Eelbrain Modules
^^^^^^^^^^^^^^^^

*	``S``:  :mod:`psystats` basic statistics such as ANOVAs and pairwise tests
*	``importer``:  import datasets
*	``op``:  perform operations of datasets
*	``plot``:  different plotting functions
*	``vw``:  interactive psychophysiology viewers (currently disabled if wxmpl is not installed) 
*	``sensors``:  create sensor nets (for EEG)
