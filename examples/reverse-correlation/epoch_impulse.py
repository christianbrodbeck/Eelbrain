"""
.. _exa-impulse:

Impulse predictors
==================
Use impulses for reverse correlation with discrete events. Impulse predictors
can be used like dummy codes in a regression model.

The dataset simulates an N400 experiment.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from eelbrain import *

ds = datasets.simulate_erp()
print(ds.summary())

###############################################################################
# Categorial coding
# -----------------
# Use one impulse to code for occurrence of any word (``any_word``), and a
# second impulse to code for unpredictable words only (``cloze``):

any_word = epoch_impulse_predictor('eeg', 1, ds=ds, name='any_word')
# effect code for cloze (1 for low cloze, -1 for high cloze)
cloze_code = Var.from_dict(ds['cloze_cat'], {'high': 0, 'low': 1})
low_cloze = epoch_impulse_predictor('eeg', cloze_code, ds=ds, name='low_cloze')

# plot the predictors for each trial
plot.UTS([any_word, low_cloze], '.case')

###############################################################################
# Estimate response functions for these two predictors. Based on the coding,
# ``any_word`` reflects the response to predictable words, and ``low_cloze``
# reflects how unpredictable words differ from predictable words:

res = boosting('eeg', [any_word, low_cloze], 0, 0.5, basis=0.050, model='cloze_cat', ds=ds, partitions=2)
p = plot.TopoButterfly(res.h)
p.set_time(0.400)
