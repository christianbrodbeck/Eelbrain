# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *

ds = datasets.get_uv()

# look at data
ds.head()

# create a fixed effects model
m = ds.eval('A * B')
# look at effects
print(repr(m))
# show the model coding
print(m)

# assert that 'rm' is a random effect: look for "random=True" at the end of the
# string representation, or look at the .random attribute:
print(ds['rm'].random)

# create random effects model
m = ds.eval('A * B * rm')
print(repr(m))
