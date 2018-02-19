# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from inspect import getargspec
import re

from .. import testnd
from .._exceptions import DefinitionError


__test__ = False
TAIL_REPR = {0: '=', 1: '>', -1: '<'}


def assemble_tests(test_dict):
    "Interpret dict with test definitions"
    out = {}
    for key, params in test_dict.iteritems():
        if isinstance(params, Test):
            out[key] = params
            continue
        elif not isinstance(params, dict):
            raise TypeError("Invalid object for test definitions %s: %r" %
                            (key, params))
        params = params.copy()
        if 'stage 1' in params:
            params['stage_1'] = params.pop('stage 1')
        kind = params.pop('kind')
        if kind in TEST_CLASSES:
            out[key] = TEST_CLASSES[kind](**params)
        else:
            raise DefinitionError("Unknown test kind in test definition %s: "
                                  "%r" % (key, kind))
    return out


def tail_arg(tail):
    try:
        if tail == 0:
            return 0
        elif tail > 0:
            return 1
        else:
            return -1
    except Exception:
        raise TypeError("tail=%r; needs to be 0, -1 or 1" % (tail,))


class Test(object):
    "Baseclass for any test"
    test_kind = None
    model = None
    vars = None

    def __init__(self, desc):
        self.desc = desc

    def as_dict(self):
        raise NotImplementedError


class EvokedTest(Test):
    "Group level test applied to subject averages"
    def __init__(self, desc, model, cat=None):
        Test.__init__(self, desc)
        self.model = model
        self.cat = cat

    def make(self, y, ds, force_permutation, kwargs):
        raise NotImplementedError


class TTestRel(EvokedTest):
    "Related measures t-test"
    test_kind = 'ttest_rel'

    def __init__(self, model, c1, c0, tail=0):
        tail = tail_arg(tail)
        desc = '%s %s %s' % (c1, TAIL_REPR[tail], c0)
        EvokedTest.__init__(self, desc, model, (c1, c0))
        self.c1 = c1
        self.c0 = c0
        self.tail = tail

    def as_dict(self):
        return {'kind': self.test_kind, 'model': self.model, 'c1': self.c1,
                'c0': self.c0, 'tail': self.tail}

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.ttest_rel(
            y, self.model, self.c1, self.c0, 'subject', ds=ds, tail=self.tail,
            force_permutation=force_permutation, **kwargs)


class TContrastRel(EvokedTest):
    "T-contrast"
    test_kind = 't_contrast_rel'

    def __init__(self, model, contrast, tail=0):
        tail = tail_arg(tail)
        EvokedTest.__init__(contrast, model)
        self.contrast = contrast
        self.tail = tail

    def as_dict(self):
        return {'kind': self.test_kind, 'model': self.model,
                'contrast': self.contrast, 'tail': self.tail}

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.t_contrast_rel(
            y, self.model, self.contrast, 'subject', ds=ds, tail=self.tail,
            force_permutation=force_permutation, **kwargs)


class ANOVA(EvokedTest):
    test_kind = 'anova'

    def __init__(self, x, model=None):
        x = ''.join(x.split())
        if model is None:
            items = sorted(i.strip() for i in x.split('*'))
            model = '%'.join(i for i in items if i != 'subject')
        EvokedTest.__init__(self, x, model)
        self.x = x

    def as_dict(self):
        return {'kind': self.test_kind, 'model': self.model, 'x': self.x}

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.anova(
            y, self.x, match='subject', ds=ds,
            force_permutation=force_permutation, **kwargs)


class TwoStageTest(Test):
    "Two-stage test on epoched or evoked data"
    test_kind = 'two-stage'

    def __init__(self, stage_1, vars=None, model=None):
        Test.__init__(self, stage_1)
        self.stage_1 = stage_1
        self.vars = vars
        self.model = model

    def as_dict(self):
        return {'kind': self.test_kind, 'stage_1': self.stage_1,
                'vars': self.vars, 'model': self.model}


TEST_CLASSES = {
    'anova': ANOVA,
    'ttest_rel': TTestRel,
    't_contrast_rel': TContrastRel,
    'two-stage': TwoStageTest,
}
AGGREGATE_FUNCTIONS = ('mean', 'rms')
DATA_RE = re.compile("(source|sensor)(?:\.(%s))?$" % '|'.join(AGGREGATE_FUNCTIONS))


class TestDims(object):
    """Data shape for test

    Paremeters
    ----------
    string : str
        String describing data.
    time : bool
        Whether the base data contains a time axis.
    """
    source = None
    sensor = None

    def __init__(self, string, time=True):
        self.time = time
        substrings = string.split()
        for substring in substrings:
            m = DATA_RE.match(substring)
            if m is None:
                raise ValueError("Invalid test dimension description: %r" %
                                 (string,))
            dim, aggregate = m.groups()
            setattr(self, dim, aggregate or True)
        if sum(map(bool, (self.source, self.sensor))) != 1:
            raise ValueError("Invalid test dimension description: %r. Need "
                             "exactly one of 'sensor' or 'source'" % (string,))
        self.string = ' '.join(substrings)

        dims = []
        if self.source is True:
            dims.append('source')
        elif self.sensor is True:
            dims.append('sensor')
        if self.time is True:
            dims.append('time')
        self.dims = tuple(dims)

        # whether parc is used from subjects or from common-brain
        if self.source is True:
            self.parc_level = 'common'
        elif self.source:
            self.parc_level = 'individual'
        else:
            self.parc_level = None

    @classmethod
    def coerce(cls, obj, time=True):
        if isinstance(obj, cls):
            if bool(obj.time) == time:
                return obj
            else:
                return cls(obj.string, time)
        else:
            return cls(obj, time)

    def __repr__(self):
        return "TestDims(%r)" % (self.string,)

    def __eq__(self, other):
        if not isinstance(other, TestDims):
            return False
        return self.string == other.string and self.time == other.time


class ROITestResult(object):
    """Store samples as attribute"""

    def __init__(self, subjects, samples, n_trials_ds, merged_dist, res):
        self.subjects = subjects
        self.samples = samples
        self.n_trials_ds = n_trials_ds
        self.merged_dist = merged_dist
        self.res = res

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in
                getargspec(self.__init__).args[1:]}

    def __setstate__(self, state):
        self.__init__(**state)
