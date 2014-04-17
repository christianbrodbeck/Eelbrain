Changes
=======

.. currentmodule:: eelbrain.data

New in 0.1.7
------------

* :class:`eelbrain.gui.SelectEpochs` can now be initialized with a single 
  :class:`mne.Epochs` instance (data needs to be preloaded).
* Parameters that take :class:`NDVar` objects now also accept 
  :class:`mne.Epochs` and :class:`mne.fiff.Evoked` objects.


New in 0.1.5
------------

* :py:class:`~.plot.topo.TopoButterfly` plot: new keyboard commands (``t``, 
  ``left arrow``, ``right arrow``).
