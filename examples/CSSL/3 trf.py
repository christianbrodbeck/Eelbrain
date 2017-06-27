# skip test
"""Compute TRFs with boosting"""
from eelbrain import *


# load the data
ds = load.fiff.events('Data/R2290_HAYO_P3H_1-8-raw.fif')
ds = ds.sub("trigger == 167")
ds['meg'] = load.fiff.epochs(ds, 0, tstop=60)

# load the saved DSS
todss, fromdss = load.unpickle('Data/R2290_HAYO_P3H_DSS.pickle')

# extract the source time course for the first DSS
meg_dss1 = todss[0].dot(ds['meg'])
# average trials (trial number is represented by the 'case' dimension)
y = meg_dss1.mean('case')
# resample at 100 Hz
y = resample(y, 100)

# load the stimulus
wav = load.wav('Stimuli/P3H.wav')
# compute envelope
env = wav.envelope()
# filter with same settings as MEG data
x = filter_data(env, 1, 8)
# resample to same sampling rate as MEG data
x = resample(x, 100)
# plot stimulus
plot.UTS([[wav, env, x]])

# boosting
res = boosting(y, x, -0.1, 0.5)

# plot TRF
plot.UTS(res.h)

# compute TRF for multiple predictors (background and mix)
wav_bg = load.wav('Stimuli/English_F_Norm_One_S.wav')
x_bg = resample(filter_data(wav_bg.envelope(), 1, 8), 100)
x_bg.name = 'background'

wav_mix = wav + wav_bg
x_mix = resample(filter_data(wav_mix.envelope(), 1, 8), 100)
x_mix.name = 'mix'

res = boosting(y, [x, x_bg, x_mix], -0.1, 0.5, error='l1')

# plot kernels separately
plot.UTS(res.h, ncol=1)
# plot kernels in one plot
plot.UTS([res.h])

# convolve the kernel with the predictors to predict the MEG response
y_pred = convolve(res.h_scaled, [x, x_bg, x_mix])
y_pred.name = "y_pred"
# compare actual and predicted response
plot.UTS([[y, y_pred]])
