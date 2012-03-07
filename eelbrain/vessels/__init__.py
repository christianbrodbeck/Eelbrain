"""
Classes for storing data along with properties, mostly for plotting functions.


Properties
----------

`ndvars` keep a properties attribute dictionary.

`contours` : epoch
    an epoch specifying the contours (needs a colorspace with contours)

`proj` : str or None
    default projection for sensors

`samplingrate`: scalar
    samplingrate in Hz

`unit`: str
    the unit of measurement (used for plotting as y-label)

`ylim` : float
    for plotting: default limit for the y-axis  

`summary_func` : func
    function used to summarize the data (normally `np.mean`). Needs to take 
    `axis` kwargs.

`summary_XXX` : 
    replaces `XXX` when a summary is generated. examples: `summary_ylim`



"""

import design
import process
import load
