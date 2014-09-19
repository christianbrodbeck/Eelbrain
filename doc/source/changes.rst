Changes
=======

.. currentmodule:: eelbrain

New in 0.9
----------

* :class:`Factor` API change:  The ``rep`` argument was renamed to ``repeat``.
* :class:`~experiment.MneExperiment`: subject name patterns and eog_sns are
  now handled automatically.


New in 0.8
----------

* A new GUI application controls plots as well as the epoch selection GUI (see
  notes in the reference sections on :ref:`ref-plotting` and :ref:`ref-guis`).
* Randomization/Monte Carlo tests now seed the random state to make results replicable.


New in 0.6
----------

* New recipes for :ref:`recipe-regression`.


New in 0.5
----------

* The :mod:`eelbrain.lab` and :mod:`eelbrain.eellab` modules are deprecated.
  Everything can now me imported from :mod:`eelbrain` directly.


New in 0.4
----------

* Optimized ANOVA evaluation, support for unbalanced fixed effects models.
* `rpy2 <http://rpy.sourceforge.net>`_ interaction: :meth:`Dataset.from_r` to
  create a :class:`Dataset` from an R Data Frame, and :meth:`Dataset.to_r` to
  cerate an R Data Frame from a :class:`Dataset`.


New in 0.3
----------

* Optimized clustering for cluster permutation tests.


New in 0.2
----------

* :class:`gui.SelectEpochs` Epoch rejection GIU has a new "GA" button to plot
  the grand average of all accepted trials
* Cluster permutation tests in :mod:`testnd` use multiple cores; To disable
  multiprocessing set ``eelbrain._stats.testnd.multiprocessing = False``.


New in 0.1.7
------------

* :class:`gui.SelectEpochs` can now be initialized with a single
  :class:`mne.Epochs` instance (data needs to be preloaded).
* Parameters that take :class:`NDVar` objects now also accept
  :class:`mne.Epochs` and :class:`mne.fiff.Evoked` objects.


New in 0.1.5
------------

* :py:class:`plot.topo.TopoButterfly` plot: new keyboard commands (``t``,
  ``left arrow``, ``right arrow``).
