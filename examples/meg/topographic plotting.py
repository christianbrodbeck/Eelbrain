# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# This script can be executed from the Terminal using $ pythonw

from eelbrain import *


w = 3

# load sample data
ds = datasets.get_mne_sample(sub="modality=='A'", sns=True)


# interactive plot
plot.TopoButterfly('sns', ds=ds)


# plot topography at a specific time point
plot.Topomap("sns.sub(time=.07)", ds=ds)

# plot topography at a specific time point by condition
plot.Topomap("sns.sub(time=.07)", 'side', ds=ds)

# plot topography at a specific time point for one condition;
# clip with a circular outline
plot.Topomap("sns.sub(side=='L', time=.07)", ds=ds, clip='circular')

# plot topography with a specific projection
plot.Topomap("sns.sub(time=.07)", proj='back', ds=ds)

# same but without clipping the map
plot.Topomap("sns.sub(time=.07)", proj='back', clip=False, ds=ds)


# mark sensors (all sensors with z coordinate < 0
y = ds["sns"].sub(time=.07).mean('case')
mark = ds['sns'].sensor.z < 0
plot.Topomap(y, mark=mark)

# different cmap, and manually adjusted head outline
plot.Topomap(y, head_radius=(0.3, 0.33), head_pos=0.02, cmap='jet')

# sensor labels
plot.SensorMap(y, mark=mark, head_radius=(0.3, 0.33), head_pos=0.02,
               labels='name')

# sensor map with labels
p = plot.SensorMap(y, mark=mark, w=9, head_radius=(0.3, 0.33), head_pos=0.02)
# move overlapping labels apart
p.separate_labels()


# run the GUI if the script is executed from the shell
if __name__ == '__main__':
    gui.run()
