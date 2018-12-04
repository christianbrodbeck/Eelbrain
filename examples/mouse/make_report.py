# skip test: data unavailable
"""Generate group level result resports for the cat-mouse experiment

This script generates HTML reports based on the experiment defined in
``mouse.py``.
"""
from mouse import e


e.set(epoch='target', inv='fixed-3-dSPM')
e.make_report('surprise', mask='frontotemporal-lh', pmin=0.05, baseline=False, tstart=0.3, tstop=0.5, raw='1-40')
e.make_report('surprise', mask='frontotemporal-lh', pmin=0.05, baseline=False, tstart=0.3, tstop=0.5, raw='fastica1-40')

e.make_report('surprise', mask='cortex', pmin=0.05, baseline=False, tstart=0.3, tstop=0.5, raw='fastica1-40')


# e.set(epoch='target', inv='free-3-dSPM')
# e.make_report('surprise', mask='frontotemporal-lh', pmin=0.05, baseline=False, tstart=0.3, tstop=0.5, raw='fastica1-40')
# e.make_report('surprise', mask='frontotemporal-lh', pmin=0.05, baseline=True, tstart=0.3, tstop=0.5, raw='fastica1-40')

# e.make_report('surprise', mask='cortex', pmin=0.05, sns_baseline=False, tstart=0.3, tstop=0.5)
# e.make_report('surprise', mask='cortex', pmin=0.05, sns_baseline=False, tstart=0.3, tstop=0.5, raw='fastica1-40')
# e.make_report('surprise', mask='cortex', pmin=0.05, sns_baseline=False, tstart=0.3, tstop=0.5, raw='fastica1-40', inv='free-3-dSPM')
