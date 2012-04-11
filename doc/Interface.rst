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


Display
^^^^^^^

* ``cmd`` - ``shift`` - ``L``: Show/hide line numbers. 
* Font:

	* ``ctrl`` - ``]``: Increase font size.
	* ``ctrl`` - ``[``: Decrease font size.
	* ``ctrl`` - ``=``: Default font size.

* Plotting

    * ``ctrl`` - ``p``: Draw plots (``pyplot.draw()``)


Entering Text
^^^^^^^^^^^^^

These shortcuts are relevant when entering text (in the shell or an editor)

* ``F1``: Open help for the command under the caret; in the help viewer the whole 
	object name must be selected.

* Moving the caret (cursor):

	* ``home``:  Go to the beginning of the command or line.
	* ``shift`` - ``Home``:  Select to the beginning of the command or line.
	* ``shift`` - ``End``:  Select to the end of the line.
	* ``end``:  Go to the end of the line.
	* ``ctrl`` - ``F``:  Search 
	* ``F3``:  Search next
	* ``ctrl`` - ``H``:  "Hide" lines containing selection / "unhide"

These work only in the Shell:

* History:

	* ``ctrl`` - ``Up Arrow``:  Retrieve previous history item.
	* ``ctrl`` - ``Down Arrow``:  Retrieve next history item.
	* ``shift`` - ``Up Arrow``:  Insert previous history item.
	* ``shift`` - ``Down Arrow``:  Insert next history item.
	* ``F8``:  Command-completion of history item. (Type a few characters of a previous 
	  command and press F8.)

* Auto Completion:

	* ``ctrl``-``Space``: Show auto completion.
	* ``ctrl``-``Alt``-``Space``: Show call tip.
	* ``shift`` - ``Enter``: Complete Text from history.

* Multi-Line Editing:

	* ``ctrl`` - ``Enter``: Insert new line into multiline command.
	* ``F12``: Turn "free-edit" mode on/off.


Shell
^^^^^

* Copying Text:

	* ``ctrl`` - ``d``:  Copy the selected commands to the topmost editor window
	* ``ctrl`` - ``c``:  Copy selected text without prompts.
	* ``ctrl`` - ``Shift`` - ``C``:  Copy selected text, retaining prompts.
	* ``ctrl`` - ``X``:  Cut selected text.
	* ``ctrl`` - ``V``:  Paste from clipboard.
	* ``ctrl`` - ``Shift`` - ``V``:  Paste and run multiple commands from clipboard.


Editor
^^^^^^

* ``ctrl`` - ``/``:  Comment or uncomment selected lines
* ``ctrl`` - ``s``:  Save current document
* ``alt`` - ``arrow (up/down)``:  Move current line up or down (!!! uses copy-paste)
* Executing Code:

	* ``ctrl`` - ``r``: Save the script and execute the whole script from disk. 
	* ``ctrl`` - ``e``: Execute the selection. 


Shell
-----

Functions
^^^^^^^^^

The following commands are available in the shell in addition to normal Python
commands. For more information, use help(command):

.. py:function:: attach(dictionary)

    Updates the global namespace with ``dictionary``, as can be shown with
    a locally defined dictionary::
    
        >>> a
        Traceback (most recent call last):
             File "<input>", line 1, in <module>
           NameError: name 'a' is not defined
           
        >>> attach({'a': 'something'})
        attached: ['a']
        >>> a
        'something'

    Many dictionary-like Eelbrain objects can be attached like that for 
    convenient access, for example: experiment.variables, datasets. The wxterm
    shell will keep track of any attached variables and
    :py:func:`detach` will remove any variables that were attached using 
    this function from the global namespace.  
	 

.. py:function:: clear()

	clear the shell

.. py:function:: copy(object)

	copy str(object) to the clipboard

.. py:function:: detach()

    remove from the global namespace any variables that were added to it 
    using the :py:func:`attach` function.

.. py:function:: help([object])

	retrieve help for any object 

.. py:function:: loadtable([filename])

	load a table from a file

.. py:function:: printdict(dictionary)

	prints a more readable representation for complex dictionaries.

.. py:function:: table([list])

	open a simple table editor. Can create a table from a 2 dimensional list as argument


Startup Script
^^^^^^^^^^^^^^

Through the menu Eelbrain->Preferences..., a ``dataDir`` can be set. If this 
dataDir contains a Python script named ``'startup'`` (note: no extension), this
script is executed every time the shell starts up (this is a feature of the
:py:class:`wx.py.shell.ShellFrame <http://www.wxpython.org/docs/api/wx.py.shell.ShellFrame-class.html>`). 

