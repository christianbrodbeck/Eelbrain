'''
Created on Jun 11, 2012

@author: christianmbrodbeck
'''
import numpy as np
import matplotlib.pyplot as plt

import sensors as sens_module
import eelbrain.eellab as E




# sensors
locs = np.array([[-1.0, 0.0, 0.0],
                 [ 0.0, 1.0, 0.0],
                 [ 1.0, 0.0, 0.0],
                 [ 0.0,-1.0, 0.0],
                 [ 0.0, 0.0, 1.0]])
sensor = sens_module.sensor_net(locs, name='testnet')

# time
T = E.var(np.arange(-.2, .8, .01), name='time')

# simulated data
Y = np.random.normal(0,1,(20,5,len(T)))
for i in xrange(10):
    phi = np.random.uniform(0,2*np.pi, 1)
    x = np.sin(10 * 2 * np.pi * (T.x + phi))
    x *= np.hanning(len(T))
    Y[i,0] += x

dims = ('case', sensor, T)
Y = E.ndvar(Y, dims=dims, name='Y', properties={'samplingrate':100})
condition = E.factor(['alpha']*10 + ['control']*10, name='condition')
ds = E.dataset(Y, condition)

def plot():
    plt.subplot(411)
    plt.imshow(Y.x[:,0,:])
    
    tt = E.testnd.ttest('Y', X='condition', c1='alpha', c2='control', ds=ds)
    E.plot.topo.array(tt)

#E.process.time_frequency(ds, source='Y', frequency=range(4, 20), 
#                       downsample=1, amplify=1, use_fft=True)
#
#Yalpha = ds['Ypower'].subdata(frequency=10)
#tt = E.testnd.ttest(ds, Yalpha, X='condition', c1='alpha', c2='control')
#p = E.plot.topo.array(tt.all)

