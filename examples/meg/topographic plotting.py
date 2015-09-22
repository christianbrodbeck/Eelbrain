# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *


w = 3

# load sample data
ds = datasets.get_mne_sample(sub="modality=='A'", sns=True)


# plot topography at a specific time point
plot.Topomap("sns.sub(time=.07)", ds=ds, method='mne')

# plot topography at a specific time point by condition
plot.Topomap("sns.sub(time=.07)", 'side', ds=ds, method='mne')

# plot topography at a specific time point for one condition
plot.Topomap("sns.sub(side=='L', time=.07)", ds=ds, method='mne')

# plot topography with a specific projection
plot.Topomap("sns.sub(time=.07)", proj='back', ds=ds, method='mne')

# mark sensors (all sensors with z coordinate < 0
y = ds["sns"].sub(time=.07).mean('case')
mark = ds['sns'].sensor.z < 0
plot.Topomap(y, sensorlabels=False, mark=mark, method='mne', res=500)

# non-circular projection, different cmap
plot.Topomap(y, sensorlabels=False, interpolation='bilinear', res=500,
             head_radius=(0.3, 0.33), head_pos=0.02, cmap='jet')

# sensor map alone, without labels
plot.SensorMap(y, 'none', mark=mark, head_radius=(0.3, 0.33), head_pos=0.02)

# sensor map with labels
p = plot.SensorMap(y, mark=mark, w=9, head_radius=(0.3, 0.33), head_pos=0.02)
p.separate_labels()


# run the GUI if the script is executed from the shell
if __name__ == '__main__':
    gui.run()
