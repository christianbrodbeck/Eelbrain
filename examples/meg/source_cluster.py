'''
Example performes a permutation cluster test on source space data and creates
an HTML file describing he output.
'''

import eelbrain.lab as e


# create an HTML report in which to document results
report = e.Report("MNE Sample Dataset", author="Mr. Beans")
section = report.add_section("Info")
section.append("Analysis of the influence of side of stimulation with auditory "
               "stimuli in the MNE sample dataset.")

'''
use the sample dataset loader to load source space data for the mne samples
dataset. Load only auditory data.
'''
ds = e.datasets.get_mne_sample(-0.1, 0.4, (None, 0), src='ico',
                               sub="modality == 'A'")

# Add a table with trial numbers to the report
freqs = e.table.frequencies('side', ds=ds)
freq_table = freqs.as_table()
section.add_figure("Number of trials in each condition", content=freq_table)


'''
perform t-test for side of stimulation (only do 100 permutations here to save
time. For a serious test set samples=10000).
'''
res = e.testnd.ttest_ind('src', 'side', 'L', 'R', ds=ds,
                         samples=100,  # number of permutations
                         pmin=0.05,  # threshold for clusters (uncorrected p-value)
                         tstart=0.05,  # start of the time window of interest
                         tstop=0.2,  # stop of the time window of interest
                         mintime=0.02,  # minimum duration for clusters
                         minsource=10)  # minimum number of sources for clusters

# sort clusters according to their p-value
res.clusters.sort("p")

# retrieve the first cluster
c_0 = res.clusters[0, 'cluster']
p = res.clusters[0, 'p']

# add a section for the first cluster to the report
section = report.add_section("Cluster 1, p=%s" % p)

# plot the extent
c_extent = c_0.sum('time')
plt_extent = e.plot.brain.cluster(c_extent)
# add to report (plot.brain plots can be captured using plot.brain.image())
image = e.plot.brain.image(plt_extent, "cluster 0 extent.png", alt=None,
                           close=False)
section.add_image_figure(image, "Extent of the largest cluster, p=%s" % p)
plt_extent.close()

# extract and analyze the value in the cluster in each trial
index = c_0 != 0
c_value = ds['src'].sum(index)
# index is a boolean NDVar over space and time, so here we are summing in the
# whole spatio-temporal cluster
plt_box = e.plot.uv.boxplot(c_value, 'side', ds=ds)
pw_table = e.test.pairwise(c_value, 'side', ds=ds)
print pw_table
# add to report (create an image from an eelbrain matplotlib plot with .image())
# use PNG because SVG boxplots do not render correctly in Safari
image = plt_box.image('image.png')
figure = section.add_figure("Cluster value")
figure.append(image)
figure.append(pw_table)

# extract and analyze the time course in the cluster
index = c_extent != 0
c_timecourse = ds['src'].sum(index)
# c_extent is a boolean NDVar over space only, so here we are summing over the
# spatial extent of the cluster for every time point but keep the time dimension
plt_tc = e.plot.UTSStat(c_timecourse, 'side', ds=ds, clusters=res.clusters[:1])
# add to report
image = plt_tc.image()
section.add_image_figure(image, "Time course of the average in the largest "
                         "cluster extent")

# save the report
report.save_html("report.html")
