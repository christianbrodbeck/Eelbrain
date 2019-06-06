"""
Two-stage test
==============

.. currentmodule:: eelbrain

When trials are associated with continuous predictor variables, averaging is
often a poor solution that loses part of the data. In such cases, a two-stage
design can be employed that allows using the continuous predictor variable to
test hypotheses at the group level. A two-stage analysis involves:

 - Stage 1: fit a regression model to each individual subject's data
 - Stage 2: test regression coefficients at the group level

The example uses simulated data meant to vaguely resemble data from an N400
experiment, but not intended as a physiologically realistic simulation.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# sphinx_gallery_thumbnail_number = 1
from eelbrain import *

###############################################################################
# Generate simulated data: each function call to :func:`datasets.simulate_erp`
# generates a dataset for one subject. For each subject, a regression model
# is fit using cloze probability as continuous predictor variables.
lms = []
for subject in range(10):
    # generate data for one subject
    ds = datasets.simulate_erp(seed=subject)
    # Re-reference EEG data
    ds['eeg'] -= ds['eeg'].mean(sensor=['LPA', 'RPA'])
    # Fit stage 1 model
    lm = testnd.LM('eeg', 'cloze', ds)
    lms.append(lm)

# Collect single-subject models for group analysis
stage2 = testnd.LMGroup(lms)

###############################################################################
# The :class:`testnd.LMGroup` object allows quick access to t-tests of
# individual regression coefficients.
res = stage2.column_ttest('cloze', pmin=0.05, tstart=0.100, tstop=0.600)
print(res.find_clusters(0.05))
p = plot.TopoButterfly(res)
p.set_time(0.400)

###############################################################################
# Since this is a regression model, it also contains an intercept coefficient
# reflecting signal deflections shared in all trials.
res = stage2.column_ttest('intercept', pmin=0.05)
p = plot.TopoButterfly(res)
p.set_time(0.120)
