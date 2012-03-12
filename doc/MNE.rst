Analysis with MNE
=================

Preliminaries
-------------

Directory structure: MNE manual (p. 20) says that ``SUBJECTS_DIR`` 
is the MRI folder and ``$SUBJECTS_DIR/$SUBJECT/bem/msh-7-src.fif`` 
is referred to as 
bem/msh-7-src.fif. This implies the following orgnizaiton::

    experiment/
               mri/
                   R0001
                   R0002
                   ...
               meg/
                   R0001
                   R0002
                   ...


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

