Analysis with MNE
=================

Preliminaries
-------------

.. MNE manual (p. 20) says that ``SUBJECTS_DIR`` 
	is the MRI folder and ``$SUBJECTS_DIR/$SUBJECT/bem/msh-7-src.fif`` 
	is referred to as 
	bem/msh-7-src.fif. This implies the following orgnizaniton


The recommended directory structure to work with mne is::

    experiment/
               mri/
                   R0001/
                   R0002/
                   ...
               meg/
                   R0001/
                         data/                   	     
                   	     parameters/
                   	     raw/
                   R0002/
                   ...

Where ``R0001`` etc. are subject identifiers.

Workflow:

 #. Use :func:`eelbrain.utils.mne_link.kit2fiff` to convert files to the fiff 
    format.
 #. Load the data using the :mod:`eelbrain.vessels.load`
 


Converting KIT data to a fiff file
----------------------------------

.. Expects the following files
   experiment/
               meg/
                   subject/
                   		   

	

.. autofunction:: eelbrain.utils.mne_link.kit2fiff



Computing the Forward Solution
------------------------------

