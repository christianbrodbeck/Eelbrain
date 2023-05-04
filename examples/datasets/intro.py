"""
.. _exa-intro:
.. currentmodule:: eelbrain

Introduction
============

Data are represented with there primary data-objects:

* :class:`Factor` for categorial variables
* :class:`Var` for scalar variables
* :class:`NDVar` for multidimensional data (e.g. a variable measured at
  different time points)

Multiple variables belonging to the same dataset can be grouped in a
:class:`Dataset` object.


Factor
======

A :class:`Factor` is a container for one-dimensional, categorial data: Each
case is described by a string label. The most obvious way to initialize a
:class:`Factor` is a list of strings:

"""
# sphinx_gallery_thumbnail_number = 5
from eelbrain import *

a = Factor(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], name='A')
print(a)

###############################################################################
# Since Factor initialization simply iterates over the given data, the
# same Factor could be initialized with:

a = Factor('aaaabbbb', name='A')
print(a)

###############################################################################
# There are other shortcuts to initialize factors  (see also
# the :class:`Factor` class documentation):

a = Factor(['a', 'b', 'c'], repeat=4, name='A')
print(a)

###############################################################################
# Indexing works like for arrays:

print(a[0])
print(a[0:6])

###############################################################################
# All values present in a :class:`Factor` are accessible in its
# :attr:`Factor.cells` attribute:

print(a.cells)

###############################################################################
# Based on the Factor's cell values, boolean indexes can be generated:

print(a == 'a')
print(a.isany('a', 'b'))
print(a.isnot('a', 'b'))

###############################################################################
# Interaction effects can be constructed from multiple factors with the ``%``
# operator:

b = Factor(['d', 'e'], repeat=2, tile=3, name='B')
print(b)
i = a % b
print(i)

###############################################################################
# Interaction effects are in many ways interchangeable with factors in places
# where a categorial model is required:

print(i.cells)
print(i == ('a', 'd'))

###############################################################################
# Var
# ===
#
# The :class:`Var` class is a container for one-dimensional
# :py:class:`numpy.ndarray`:

y = Var([1, 2, 3, 4, 5, 6])
print(y)

###############################################################################
# Indexing works as for factors

print(y[5])
print(y[2:])

###############################################################################
# Many array operations can be performed on the object directly

print(y + 1)

###############################################################################
# For any more complex operations the corresponding :py:class:`numpy.ndarray`
# can be retrieved in the :attr:`Var.x` attribute:

print(y.x)

###############################################################################
# .. Note::
#     The :attr:`Var.x` attribute is not intended to be replaced; rather, a new
#     :class:`Var` object should be created for a new array.
#
#
# NDVar
# =====
#
# :class:`NDVar` objects are containers for multidimensional data, and manage the
# description of the dimensions along with the data. :class:`NDVar` objects are
# often not constructed from scratch but imported from existing data. For
# example, :mod:`mne` source estimates can be imported with
# :func:`load.mne.stc_ndvar`. As an example, consider data from a simulated EEG
# experiment:

ds = datasets.simulate_erp()
eeg = ds['eeg']
print(eeg)

###############################################################################
# This representation shows that ``eeg`` contains 80 trials of data (cases),
# with 140 time points and 35 EEG sensors. Since ``eeg`` contains information
# on the dimensions like sensor locations, plotting functions can take
# advantage of that:

p = plot.TopoButterfly(eeg)
p.set_time(0.400)

###############################################################################
# :class:`NDVar` offer functionality similar to :class:`numpy.ndarray`, but
# take into account the properties of the dimensions. For example, through the
# :meth:`NDVar.sub` method, indexing can be done using meaningful descriptions,
# such as indexing a time slice in seconds:

eeg_400 = eeg.sub(time=0.400)
plot.Topomap(eeg_400)

###############################################################################
# Several methods allow aggregating data, for example an RMS over sensor:

eeg_rms = eeg.rms('sensor')
print(eeg_rms)
plot.UTSStat(eeg_rms)

###############################################################################
# Or a mean in a time window:

eeg_400 = eeg.mean(time=(0.350, 0.450))
plot.Topomap(eeg_400)

###############################################################################
# As with a :class:`Var`, the corresponding :class:`numpy.ndarray` can always be
# accessed as array. The :meth:`NDVar.get_data` method allows retrieving the
# data while being explicit about which axis represents which dimension:

array = eeg_400.get_data(('case', 'sensor'))
print(array.shape)

###############################################################################
# :class:`NDVar` objects can be constructed directly from an array and
# corresponding dimension objects, for example:

import numpy

frequency = Scalar('frequency', [1, 2, 3, 4])
time = UTS(0, 0.01, 50)
data = numpy.random.normal(0, 1, (4, 50))
ndvar = NDVar(data, (frequency, time))
print(ndvar)

###############################################################################
# A case dimension can be added by including the bare :class:`Case` class:
#
data = numpy.random.normal(0, 1, (10, 4, 50))
ndvar = NDVar(data, (Case, frequency, time))
print(ndvar)

###############################################################################
# Dataset
# =======
#
# A :class:`Dataset` is a container for multiple variables
# (:class:`Factor`, :class:`Var` and :class:`NDVar`) that describe the same
# cases. It can be thought of as a data table with columns corresponding to 
# different variables and rows to different cases. Variables can be assigned
# as to a dictionary:

ds = Dataset()
ds['x'] = Factor('aaabbb')
ds['y'] = Var([5, 4, 6, 2, 1, 3])
print(ds)

###############################################################################
# A variable that's equal in all cases can be assigned quickly:

ds[:, 'z'] = 0.

###############################################################################
# The string representation of a :class:`Dataset` contains information
# on the variables stored in it:

# in an interactive shell this would be the output of just typing ``ds``
print(repr(ds))

###############################################################################
# ``n_cases=6`` indicates that the Dataset contains 6 cases (rows). The
# subsequent dictionary-like representation shows the keys and the types of the
# corresponding values (``F``:   :class:`Factor`, ``V``:   :class:`Var`,
# ``Vnd``: :class:`NDVar`).
#
# A more extensive summary can be printed with the :meth:`Dataset.summary`
# method:

print(ds.summary())

###############################################################################
# Indexing a Dataset with strings returns the corresponding data-objects:

print(ds['x'])

###############################################################################
# :class:`numpy.ndarray`-like indexing on the Dataset can be used to access a
# subset of cases:

print(ds[2:])

###############################################################################
# Row and column can be indexed simultaneously (in row, column order):

print(ds[2, 'x'])

###############################################################################
# Arry-based indexing also allows indexing based on the Dataset's variables:

print(ds[ds['x'] == 'a'])

###############################################################################
# Since the dataset acts as container for variable, there is a
# :meth:`Dataset.eval` method for evaluatuing code strings in the namespace
# defined by the dataset, which means that dataset variables can be invoked
# with just their name:

print(ds.eval("x == 'a'"))

###############################################################################
# Many dataset methods allow using code strings as shortcuts for expressions
# involving dataset variables, for example indexing:

print(ds.sub("x == 'a'"))

###############################################################################
# Example
# =======
#
# Below is a simple example using data objects (for more, see the
# :ref:`examples`):

y = numpy.empty(21)
y[:14] = numpy.random.normal(0, 1, 14)
y[14:] = numpy.random.normal(2, 1, 7)
ds = Dataset({
    'a': Factor('abc', 'A', repeat=7),
    'y': Var(y, 'Y'),
})
print(ds)
###############################################################################
print(table.frequencies('a', data=ds))
###############################################################################
print(test.ANOVA('y', 'a', data=ds))
###############################################################################
print(test.pairwise('y', 'a', data=ds, corr='Hochberg'))
###############################################################################
t = test.pairwise('y', 'a', data=ds, corr='Hochberg')
print(t.get_tex())
###############################################################################
plot.Boxplot('y', 'a', data=ds, title="My Boxplot", ylabel="value", corr='Hochberg')
