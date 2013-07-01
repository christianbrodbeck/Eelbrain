Changes
=======

Since 0.0.4
-----------

WxTerm interface
^^^^^^^^^^^^^^^^

* The startup script is now managed by Eelbrain and the dataDir setting has 
  been removed (See the preferences to modify the startup script).
* Command history can now be saved (see the preferences to enable/disable).
* Executing code is now handled through the new "Exec" Menu with new keyboard 
  shortcuts. In the editor, ``cmd-Enter`` when nothing is selected executes 
  the current line and moves the caret to the next line. This makes it 
  possible to step through a script line by line.


:class:`~eelbrain.vessels.experiment.mne_experiment`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: eelbrain.vessels.mne_experiment

* Ability to use manual epoch rejection instead of the automatic procedure.
  The ``epoch_rejection`` class attribute specifies how epochs should be 
  rejected.
* The ``epoch`` can now be set with ``.set(epoch=...)``, and a default 
  epoch can be specified in the :attr:`_defaults` class attribute.
* The ``'root'`` is now treated like a normal templates entry, i.e., it can
  be specified through the :attr:`_templates` and :attr:`_defaults` class 
  attributes as well as in :meth:`__init__`.
* More consistent template names:

  * all template names now use '-' instead of '_' for name-kind sequences.
  * all paths to files are called '*-file'.
  * all paths to directories are called '*-dir'.
  * all paths to directories containing subjects are called '*-sdir'.

* Per-file settings (such as bad channel definition) are now stored with custom 
  keys. ``bad_chs = e.bad_channels[e.get('badch-key')]``. By default, the key
  is just ``'{subject}'``, but if for example multiple experiments are 
  processed with the same experiment class, the key can be changed to something 
  like ``'{subject}_{experimnt}'``.


:class:`~eelbrain.vessels.data.dataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: eelbrain.vessels.data

* Idexing with sequence of strings now returns a :class:`dataset` (previously 
  a :class:`list`). Example: ``new_ds = ds['var1', 'factor3']``. This is 
  consistent with numerical indexing, such as ``new_ds = ds[:10, ('var1', 
  'factor3')]``.
* New save methods replace :meth:`dataset.export`. See :meth:`dataset.save`.
