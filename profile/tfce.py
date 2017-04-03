from eelbrain import *

configure(False)

ds = datasets.get_mne_sample(-0.1, 0.2, src='ico', sub="modality == 'A'")
res = testnd.ttest_1samp('src', ds=ds, tstart=0, tfce=True, samples=100)

# ds = datasets.get_uts()
# res = testnd.ttest_1samp('uts', ds=ds, tfce=True, samples=1000)
