from eelbrain.eellab import *

ds = datasets.get_basic()

print test.anova('Y', 'A*B', ds=ds)

plot.uv.boxplot('Y', 'A%B', ds=ds)
