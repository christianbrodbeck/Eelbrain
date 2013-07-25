"""
Operations that can be performed on segments.


StatsSegment Operations:
- corr
"""

import time, logging

import numpy as np
import scipy as sp

from ..data.test import glm

import segments as _seg
import mat




def mean(*args):
    """
    in: items as arguments or as list of arguments
    out: mean

    takes into account some special proberties of eelbrain items

    """
    if len(args) == 1:
        args = args[0]
    if len(args) == 0:
        raise ValueError("0-length sequence has no mean")
    if all([hasattr(item, 'name') for item in args]):
        new_name = "mean(%s)" % (', '.join([item.name for item in args]))
    else:
        new_name = None

    out = reduce(lambda x, y: x + y, args) / len(args)

    if new_name:
        out.name = new_name

    return out


def stack(*StatsSegments, **kwargs):
    """
    Returns a StatsSegment that contains stacked data from several
    StatsSegments (cf. with StatsSegment addition, which sums the data for each
    value on slist)

    Keyword Arguments
    ---------

    name: string
        Name for new segment. The default is the names of the input
        StatsSegments joined with '&'.


    """
    # analyse kwargs
    name = kwargs.get('name', ' & '.join(s.name for s in StatsSegments))

    s0 = StatsSegments[0]
    assert all(_seg.iscollection(s) for s in StatsSegments), "can only stack StatsSegments"
    assert all(s.svar == s0.svar for s in StatsSegments[1:]), "Segments need to have same svar"
    data = np.concatenate([s.data for s in StatsSegments], axis=-1)
    slist = [item for s in StatsSegments for item in s.slist]  # http://stackoverflow.com/questions/952914
    return s0._create_child(data, name=name, mod_props={'slist': slist})


def test(segments, parametric=True, related='auto', func=None, attr='data',
         name="{testname}"):
    """
    use func (func) or attr (str) to customize data
    (func=abs for )

    """
    # args
    if _seg.issegment(segments):
        segments = [segments]
    if related == 'auto':
        if hasattr(segments[0], 'list'):
            slist = segments[0].slist
            if all([s.slist == slist for s in segments]):
                related = True
            else:
                related = False
        else:
            related = False
    properties = segments[0].properties

    # data
    data = [eval("s.%s" % attr) for s in segments]
    if func != None:
        data = func(data)
    # test
    k = len(segments)  # number of levels
    if k == 0:
        raise ValueError("no segments provided")

    # perform test
    if parametric:  # simple tests
        if k == 1:
            statistic = 't'
            T, P = sp.stats.ttest_1samp(*data, popmean=0, axis=-1)
            test_name = '1-Sample $t$-Test'
        elif k == 2:
            statistic = 't'
            if related:
                T, P = sp.stats.ttest_rel(*data, axis=-1)
                test_name = 'Related Samples $t$-Test'
            else:
                T, P = sp.stats.ttest_ind(*data, axis=-1)
                test_name = 'Independent Samples $t$-Test'
        else:
            statistic = 'F'
            raise NotImplementedError("Use segframe for 1-way ANOVA")

    else:  # non-parametric:
        if k <= 2:
            if related:
                test_func = sp.stats.wilcoxon
                statistic = 't'
                test_name = 'Wilcoxon'
            else:
                raise NotImplementedError()
        else:
            if related:
                test_func = sp.stats.friedmanchisquare
                statistic = 'Chi**2'
                test_name = 'Friedman'
            else:
                raise NotImplementedError()

        shape = segments[0].shape[:-1]
        # function to apply stat test to array
        def testField(*args):
            """
            will be executed for a gird, args will contain coordinates of shape
            assumes that subjects are on last axis

            """
            rargs = [np.ravel(a) for a in args]
            T = []
            P = []
            for indexes in zip(*rargs):
                index = tuple([slice(int(i), int(i + 1)) for i in indexes] + [slice(0, None)])
                testArgs = tuple([ d[index].ravel() for d in data])
                t, p = test_func(*testArgs)
                T.append(t)
                P.append(p)
            P = np.array(P).reshape(args[0].shape)
            T = np.array(T).reshape(args[0].shape)
            return T, P
        T, P = np.fromfunction(testField, shape)

    # Direction of the effect
    if len(segments) == 2:
        direction = np.sign(eval("segments[0].%s.mean(-1) - segments[1].%s.mean(-1)" % (attr, attr)))
    elif len(segments) == 1:
        direction = np.sign(eval("segments[0].%s.mean(-1)" % attr))
    else:
        direction = None

    # create test_segment
    name = name.format(testname=test_name)
    test_seg = _seg.StatsTestSegment(properties, T, P, dir=direction, name=name,
                                     statistic=statistic)
    return test_seg


def anova_persample(segments, X):
    """
    Perform multiple ANOVAs for segments. X must include measurements, but not
    time or sensor.

    returns a list with a StatsTestSegment for each effect for which p-values
    exist

    """
    t0 = time.time()
    lm = glm.lm_fitter(X)
    ndim = segments[0].data.ndim - 1
    Y = np.concatenate([s.data for s in segments], axis=ndim)

    # create Test Segments
    p_fields = []
    properties = segments[0].properties
    for name, F, P in lm.map(Y):
        new = _seg.StatsTestSegment(properties, F, P, statistic='F', name=name)
        p_fields.append(new)

    # log time
    t = time.time() - t0
    msg = "ANOVA p field estimation took %s min %s s"
    logging.debug(msg % (int(t / 60), int(t % 60)))

    return p_fields





def corr(stats_seg, var, sensor=0):
    """
    parameters
    ----------
    stats_seg: StatsSegment

    var: - variable Parasite; hosts need to be
         - another stats_segment
    returns segment containing correlation of stats_seg with var
    """
    assert isinstance(stats_seg, _seg.StatsSegment)
    data = stats_seg.data[:, sensor, :]
    if isinstance(var, _seg.StatsSegment):
        assert np.all(stats_seg.slist == var.slist)
        data2 = var.data[:, sensor, :]
        assert data.shape == data2.shape
        corr = [np.corrcoef(d1, d2)[0, 1] for d1, d2 in zip(data, data2)]
    else:
        if len(var.hosts) > 1:
            raise NotImplementedError()
        if var.hosts[0].name != stats_seg['svar']:
            raise ValueError("var host '%s' != svar '%s'" % \
                             (var.hosts[0].name, stats_seg['svar']))

        corr = []  # the correlation coefficients
        slist = [(s,) for s in stats_seg['slist']]
        x = var.values_from_sourceindexes(slist)
        for i in range(data.shape[0]):
            c = np.corrcoef(data[i], x, rowvar=False)[0, 1]
            corr.append(c)
    R = np.array(corr)[:, None]
    # P (http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference)
    n = len(stats_seg.slist)
    Ts = np.abs(R) / np.sqrt((1 - R ** 2) / (n - 2))
    df = n - 2
    Ps = 2 * sp.stats.t.sf(Ts, df)

    kwargs = dict(name='r(%s, %s)' % (stats_seg.name, var.name),
                parameterName='r')
    out = _seg.StatsTestSegment(stats_seg.properties, R, Ps, dir=np.sign(R),
                                **kwargs)
    out['df'] = df
    return out


def rms(segment, ROI=None, ROIs=None, sensor=None):
    """
    Root Mean Square
    ----------------
    any of the following kwargs
     - False -> all sensors are used (=global field power; GFP)
     - scalar -> one sensor
     - ROI (list of sensors)
     - list of ROIs (lists of list of sensors) -> extract rms for each ROI
    """
    data = segment.data
    if ROIs is not None:
        n = len(ROIs)
        old_shape = data.shape
        shape = (old_shape[0], n) + old_shape[2:]
        out = np.empty(shape)
        for i, ROI in enumerate(ROIs):
            out[:, [i]] = mat.RMS(data[:, ROI])
    else:
        if ROI is not None:
            data = data[:, ROI]
        elif sensor is not None:
            data = data[:, [sensor]]

        out = mat.RMS(data)

    # create segment
    out_segment = segment._create_child(out, rm_props=['sensors'],
                                        mod_props={})
    return out_segment

