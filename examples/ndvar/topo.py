import numpy as np
from eelbrain.eellab import *
import eelbrain.vessels.colorspaces as cs
from eelbrain.vessels.sensors import sensor_net


"""
Create simulated data with shape 
(2 conditions * 15 subjects, 5 sensors, len(T) time points)

"""
# create the time dimension
T = var(np.arange(-.2, .8, .01), name='time')

# random data
x = np.random.normal(0,1,(30,5,len(T)))
# add an effect to the random data
x[15:,:3,20:40] += np.hanning(20) * 2

# create the sensor dimension from 5 sensor locations in 3d space
sensor = sensor_net([[0,0,0],[1,0,0],[0,-1,0],[-1,0,0],[0,1,0]], 
                    name='testnet', transform_2d=None)

# combine all these into the ndvar
Y = ndvar(x, dims=('case', sensor, T), name='Y', 
          properties={'samplingrate':100, 'colorspace':cs.get_MEG(2)})



# To describe the cases ('case' dimension), create a condition and a subject factor
A = factor(['a0', 'a1'], rep=15, name='A')
subject = factor(xrange(15), tile=2, random=True, name='subject')


if 01: # plot topographically an uncorrected t-test
    plot.topo.array(testnd.ttest(Y, A, match=subject))

if 01: # and a butterfly plot
    plot.topo.butterfly(testnd.ttest(Y, A, match=subject))
