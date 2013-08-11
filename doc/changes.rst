Changes
=======

Since 0.0.4
-----------

WxTerm interface
^^^^^^^^^^^^^^^^

* The startup script is now managed by Eelbrain and the dataDir setting has 
  been removed (See the preferences to modify the startup script).
* Command history is now saved automatically (see the preferences to 
  enable/disable). The history for saved sessions can be opened from the 
  *File/History* menu.
* Executing code is now handled through the new "Exec" Menu with new keyboard 
  shortcuts. In the editor, ``cmd-Enter`` when nothing is selected executes 
  the current line and moves the caret to the next line. This makes it 
  possible to step through a script line by line.


:class:`~eelbrain.data.Dataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: eelbrain.vessels.data

* Idexing with sequence of strings now returns a :class:`Dataset` (previously 
  a :class:`list`). Example: ``new_ds = ds['var1', 'factor3']``. This is 
  consistent with numerical indexing, such as ``new_ds = ds[:10, ('var1', 
  'factor3')]``.
* New save methods replace :meth:`Dataset.export`. See :meth:`Dataset.save`.


Plotting Functions
^^^^^^^^^^^^^^^^^^

* New uniform layout arguments for all plots with subplots of the same type
  (see :ref:`plotting-general`).
  