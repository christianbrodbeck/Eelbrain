.. _mne-coreg-info:

**************
Coregistration
**************

.. note:: The plotting functions mentioned are all in 
   :mod:`eelbrain.plot.coreg`.


MRI-Head Coregistration
=======================

Subjects with MRI
-----------------

#. Use :class:`~eelbrain.plot.coreg.set_nasion` to define the location of the
   nasion on the subject's MRI. Save a fiducials file (dummy values are saved
   for the auricular points).
#. Use :class:`~eelbrain.plot.coreg.mri_head_viewer`.
   `s_from` is the name of the subject (which should be identical for the MRI
   and the MEG subject).
   The position of head and MRI are initially aligned using the nasion.
   Use "Fit no scale" to fit the MRI to the head using rotation only.
   Modify the resulting parameters to achieve a satisfying fit:
   **Nasion**: Shift the position of the nasion alignment.
   **Rotation**: Rotation parameters applied with the nasion as center.
   **Fit ...**: at any time you can make a new fit, taking the current parameters
   as the starting point.
#. Once a satisfactory coregistration is achieved, hit "Save trans" to save
   the trans file.

Subjects without MRI
--------------------

#. Use :class:`~eelbrain.plot.coreg.mri_head_viewer`;
   `s_from` is the MRI to use (usually `'fsaverage'`), `s_to` is the subject
   for which the MRI will be used.
   The position of head and MRI are initially aligned using the nasion.
#. Use "Fit scale" to fit the MRI to the head using rotation and scaling.
   Modify the resulting parameters to achieve a satisfying fit:
   **Nasion**: Shift the position of the nasion alignment.
   **Scale**: The scale parameters applies with the nasion as center.
   **Shrink**: Shrink the MRI (affects scale on all axes) to compensate for
   e.g. an inflated digitizer head shape.
   **Rotation**: Rotation parameters applied with the nasion as center.
   **Fit ...**: at any time you can make a new fit, taking the current parameters
   as the starting point.
#. Once a satisfactory coregistration is achieved, hit "Save" to save the MRI
   as well as the trans file.


Device-Head Coregistration
==========================

#. Use :class:`~eelbrain.plot.coreg.dev_head_viewer` to examine the
   coregistration (the initially displayed coregistration is the one contained
   in the raw file).
   If the coregistration is okay, stop here.
#. If necessary, re-fit the device-head coregistration (tick the "refit" box)
   and adjust the fit by excluding some of the marker points (0 through 4).
   When done, hit "save" to save a modified raw file.


Check the Coregistration
========================

:class:`~eelbrain.plot.coreg.dev_mri` plots a mayavi figure (without
interface) that can be used to check that the coregistration is ok.
