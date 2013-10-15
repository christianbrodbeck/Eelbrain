from eelbrain.eellab import *

ds = datasets.get_uv()

print test.anova('fltvar', 'A*B', ds=ds)

plot.uv.boxplot('fltvar', 'A%B', ds=ds)
