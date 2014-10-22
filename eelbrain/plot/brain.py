"""
Plot :class:`NDVar` objects containing source estimates with mayavi/pysurfer.

.. autosummary::
   :toctree: generated

   activation
   cluster
   dspm
   stat
   surfer_brain

Other plotting functions:

.. autosummary::
   :toctree: generated

   annot

Functions that can be applied to the :class:`surfer.Brain` instances that are
returned by the plotting functions:

.. autosummary::
   :toctree: generated

    bin_table
    copy
    image

"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from ._brain import (activation, cluster, dspm, stat, surfer_brain, annot,
                     bin_table, copy, image)
