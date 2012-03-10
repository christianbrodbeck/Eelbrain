"""
op
==

Parent module for dataset operations. Operations are grouped into the 
following sub-modules:

base:
    basic operations like cropping segments
    
evt: 
    operations on event datasets

filt: 
    filtering

physio: 
    psychophysiology

spec:
    spectral analysis


"""

import datasets_op_base as base
import datasets_op_physio as physio
import datasets_op_spec as spec
import datasets_op_filt as filt
#import datasets_op_eeg as eeg

import datasets_op_events as evt
