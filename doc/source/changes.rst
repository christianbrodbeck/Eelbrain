Changes
=======

.. currentmodule:: eelbrain

New in 0.12
-----------

* API:  :class:`plot.SensorMap2d` was renamed to :class:`plot.SensorMap`.
* :class:`~experiment.MneExperiment`:

  - New epoch parameter ``'n_cases'``:  raise an error if an epoch definition
    does not yield expected number of trials.


New in 0.11
-----------

* :class:`~experiment.MneExperiment`:

  - Change in the way the covariance matrix
    is defined:  The epoch for the covariance matrix should be specified in
    ``MneExperiment.epochs['cov']``. The regularization is no longer part of
    :meth:`~experiment.MneExperiment.set_inv`, but is instead set with
    ``MneExperiment.set(cov='reg')`` or ``MneExperiment.set(cov='noreg')``.
  - New option ``cov='bestreg'`` automatically selects the regularization
    parameter for each subejct.

* :meth:`Var.as_factor` allows more efficient labeling when multiple values
  share the same label.
* API: Previously :func:`plot.configure_backend` is now :func:`plot.configure`


New in 0.10
-----------

* Tools for generating colors for categories (see :ref:`ref-plotting`).
* Plots now all largely respect matplotlib rc-parameters (see
  `Customizing Matplotlib <http://matplotlib.org/users/customizing.html>`_).
* Fixed an issue in the :mod:`testnd` module that could affect permutation based
  p-values when multiprocessing was used.


New in 0.9
----------

* :class:`Factor` API change:  The ``rep`` argument was renamed to ``repeat``.
* T-values for regression coefficients through :meth:`NDVar.ols_t`.
* :class:`~experiment.MneExperiment`: subject name patterns and eog_sns are
  now handled automatically.
* :class:`~plot.uts.UTSStat` and :class:`~plot.uv.Barplot` plots can use pooled
  error for variability estimates (on by default for related measures designs,
  can be turned off using the ``pool_error`` argument).

  - API: for consistency, the argument to specify the kind of error to plot
    changed to ``error`` in both plots.


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
