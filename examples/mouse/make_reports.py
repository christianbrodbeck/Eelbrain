# skip test: data unavailable
"""Generate group level result resports for the Mouse experiment

This script generates HTML reports based on the experiment defined in
``mouse.py``.
"""
from mouse import e


e.set(epoch='target', inv='fixed-3-dSPM')
e.make_report('surprise', mask='frontotemporal-lh', pmin=0.05, baseline=False, tstart=0.3, tstop=0.5, raw='1-40')
e.make_report('surprise', mask='frontotemporal-lh', pmin=0.05, baseline=False, tstart=0.3, tstop=0.5, raw='fastica1-40')
