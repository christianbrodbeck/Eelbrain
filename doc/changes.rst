Changes
=======

Since 0.0.4
-----------

* :class:`~eelbrain.vessels.experiment.mne_experiment`

	* Ability to use manual epoch rejection instead of the automatic procedure.
	  The ``epoch_rejection`` class attribute specifies how epochs should be 
	  rejected.
	* The ``epoch`` can now be set with ``.set(epoch=...)``, and a default 
	  epoch can be specified in the ``_defaults`` class attribute.
