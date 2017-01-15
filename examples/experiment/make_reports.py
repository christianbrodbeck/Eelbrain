# skip test
"""A script that creates test reports for an MneExperiment subclass

"""
from sample_experiment import SampleExperiment, ROOT


# create the experiment class instance
e = SampleExperiment(ROOT)

# Use this to send an email to e.owner when the reports are finished or the
# script raises an error
with e.notification:
    # Whole-brain test with default settings
    e.make_report('a>v', mask='lobes', pmin=0.05, tstart=0.05, tstop=0.2)

    # different inverse solution
    e.make_report('a>v', mask='lobes', pmin=0.05, tstart=0.05, tstop=0.2,
                  inv='fixed-3-dSPM')

    # test on a different epoch (using only auditory trials)
    # note that inv is still 'fixed-3-dSPM' unless it is set again
    e.make_report('left=right', mask='lobes', pmin=0.05, tstart=0.05, tstop=0.2,
                  epoch='auditory')
