import numpy as np
from eelbrain import *
# dimension objects
from eelbrain._data_obj import UTS, Sensor


"""
Create simulated data with shape 
(2 conditions * 15 subjects, 5 sensors, len(T) time points)

"""
# create the time dimension
time = UTS(-.2, .01, 100)

# random data
x = np.random.normal(0, 1, (30, 5, len(time)))
# add an effect to the random data
x[15:,:3,20:40] += np.hanning(20) * 2

# create the sensor dimension from 5 sensor locations in 3d space
sensor = Sensor([[0,0,0],[1,0,0],[0,-1,0],[-1,0,0],[0,1,0]], 
                sysname='testnet', proj2d=None)

# combine all these into the NDVar. Plotting defaults are stored in the info 
# dict:
info = {'vmax': 2.5, 'meas': 'B', 'cmap': 'xpolar', 'unit': 'pT'}
Y = NDVar(x, dims=('case', sensor, time), name='Y', info=info)



# To describe the cases ('case' dimension), create a condition and a subject Factor
A = Factor(['a0', 'a1'], repeat=15, name='A')
subject = Factor(xrange(15), tile=2, random=True, name='subject')

# uncorrected related measures t-test
res = testnd.ttest_rel(Y, A, match=subject)

# plot topographically an uncorrected t-test
plot.TopoArray(res)

# and a butterfly plot
plot.TopoButterfly(res)
