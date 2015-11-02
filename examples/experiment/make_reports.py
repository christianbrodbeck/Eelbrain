from word_experiment import WordExperiment


# create the experiment class instance
e = WordExperiment("/files")

with e.notification:
    # Whole-brain test with default settings
    e.make_report('noun>verb', mask='lobes', pmin=0.05, tstart=0.15,
                  tstop=0.35)

    # different source solution
    e.make_report('noun>verb', mask='lobes', pmin=0.05, tstart=0.15,
                  tstop=0.35, inv='fixed-3-dSPM')

    # test on a different epoch (comprised of a subset of trials)
    # note that inv is still 'fixed-3-dSPM' unless it is set again
    e.make_report('noun>verb', mask='lobes', pmin=0.05, tstart=0.15,
                  tstop=0.35, epoch='high_frequency')
