"""
.. _exa-dataset:

Dataset basics
==============
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *
import numpy

###############################################################################
# A dataset can be constructed column by column, by adding one variable after
# another:

# initialize an empty Dataset:
ds = Dataset()
# numeric values are added as Var object:
ds['y'] = Var(numpy.random.normal(0, 1, 6))
# categorical data as represented in Factors:
ds['a'] = Factor(['a', 'b', 'c'], repeat=2)
# A variable that's equal in all cases can be assigned quickly:
ds[:, 'z'] = 0.
# check the result:
ds

###############################################################################
# For larger datasets it can be more convenient to print only the first few
# cases...
ds.head()

###############################################################################
# ... or a summary of variables:
ds.summary()

###############################################################################
# An alternative way of constructing a dataset is case by case (i.e., row by
# row):

rows = []
for i in range(6):
    subject = f'S{i}'
    y = numpy.random.normal(0, 1)
    a = 'abc'[i % 3]
    rows.append([subject, y, a])
ds = Dataset.from_caselist(['subject', 'y', 'a'], rows, random='subject')
ds

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
ds

###############################################################################
table.frequencies('a', data=ds)

###############################################################################
test.ANOVA('y', 'a', data=ds)

###############################################################################
test.pairwise('y', 'a', data=ds, corr='Hochberg')

###############################################################################
t = test.pairwise('y', 'a', data=ds, corr='Hochberg')
print(t.get_tex())

###############################################################################
p = plot.Boxplot('y', 'a', data=ds, title="My Boxplot", ylabel="value", corr='Hochberg')
