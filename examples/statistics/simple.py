from eelbrain.eellab import *
import numpy as np

np.random.seed(2)

y = np.empty(21)
y[:14] = np.random.normal(0, 1, 14)
y[14:] = np.random.normal(1.5, 1, 7)

Y = var(y, 'Y')
A = factor('abc', 'A', rep=7)

print dataset(Y, A)
print test.anova(Y, A)
print test.pairwise(Y, A, corr='Hochberg')

t = test.pairwise(Y, A, corr='Hochberg')
print t.get_tex()

plot.uv.boxplot(Y, A, title="My Boxplot", ylabel="value", corr='Hochberg')
