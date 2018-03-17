# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from .._exceptions import DefinitionError
from .definitions import Definition, typed_arg


def assemble_epochs(epoch_def, epoch_default):
    epochs = {}
    secondary_epochs = []
    super_epochs = []
    collections = []
    for name, parameters in epoch_def.items():
        # filter out secondary epochs
        if 'base' in parameters:
            secondary_epochs.append((name, parameters.copy()))
        elif 'sub_epochs' in parameters:
            super_epochs.append((name, parameters))
        elif 'collect' in parameters:
            collections.append((name, parameters))
        else:
            kwargs = epoch_default.copy()
            kwargs.update(parameters)
            epochs[name] = PrimaryEpoch(name, **kwargs)

    # integrate secondary epochs (epochs with base parameter)
    while secondary_epochs:
        n_secondary_epochs = len(secondary_epochs)
        for i in range(n_secondary_epochs - 1, -1, -1):
            name, parameters = secondary_epochs[i]
            if parameters['base'] in epochs:
                parameters['base'] = epochs[parameters['base']]
                epochs[name] = SecondaryEpoch(name, **parameters)
                del secondary_epochs[i]
        if len(secondary_epochs) == n_secondary_epochs:
            raise ValueError("Invalid epoch definition: " +
                             '; '.join('Epoch %s has non-existing base '
                                       '%r.' % p for p in secondary_epochs))
    # integrate super-epochs
    for cls, arg, items in ((SuperEpoch, 'sub_epochs', super_epochs),
                            (EpochCollection, 'collect', collections)):
        epochs_ = {}
        for name, parameters in items:
            parameters = parameters.copy()
            sub_keys = parameters.pop(arg)
            try:
                sub_epochs = [epochs[n] for n in sub_keys]
            except KeyError as err:
                raise DefinitionError(
                    '%s %r %s=%s: no epoch named %r (recursive definitions are '
                    'not allowed)' % (cls.__name__, name, arg, sub_keys, err.args))
            epochs_[name] = cls(name, sub_epochs, **parameters)
        epochs.update(epochs_)
    return epochs


class Epoch(Definition):
    """Epoch definition base (non-functional baseclass)

    Parameters
    ----------
    ...
    trigger_shift : float | str
        Trigger shift applied after loading selected events. Trigger shift is
        applied for all Epoch subtypes, i.e., it combines additively for
        secondary epochs.
    """
    DICT_ATTRS = ('name', 'tmin', 'tmax', 'decim', 'baseline', 'vars',
                  'trigger_shift', 'post_baseline_trigger_shift',
                  'post_baseline_trigger_shift_min',
                  'post_baseline_trigger_shift_max')

    # to be set by subclass
    rej_file_epochs = None
    sessions = None

    def __init__(self, name, tmin=-0.1, tmax=0.6, decim=5, baseline=(None, 0),
                 vars=None, trigger_shift=0., post_baseline_trigger_shift=None,
                 post_baseline_trigger_shift_min=None,
                 post_baseline_trigger_shift_max=None, sel_epoch=None):
        if sel_epoch is not None:
            raise DefinitionError("The `sel_epoch` epoch parameter has been "
                                  "removed. Use `base` instead.")

        if (post_baseline_trigger_shift is not None and
                (post_baseline_trigger_shift_min is None or
                 post_baseline_trigger_shift_max is None)):
            raise ValueError("Epoch %s contains post_baseline_trigger_shift "
                             "but is missing post_baseline_trigger_shift_min "
                             "and/or post_baseline_trigger_shift_max" % name)

        if baseline is None:
            baseline = (None, 0)
        elif len(baseline) != 2:
            raise ValueError("Epoch baseline needs to be length 2 tuple, got "
                             "%s" % repr(baseline))
        else:
            baseline = (typed_arg(baseline[0], float),
                        typed_arg(baseline[1], float))

        if not isinstance(trigger_shift, (float, str)):
            raise TypeError("trigger_shift needs to be float or str, got %s" %
                            repr(trigger_shift))

        self.name = name
        self.tmin = typed_arg(tmin, float)
        self.tmax = typed_arg(tmax, float)
        self.decim = typed_arg(decim, int)
        self.baseline = baseline
        self.vars = vars
        self.trigger_shift = trigger_shift
        self.post_baseline_trigger_shift = post_baseline_trigger_shift
        self.post_baseline_trigger_shift_min = post_baseline_trigger_shift_min
        self.post_baseline_trigger_shift_max = post_baseline_trigger_shift_max

    def as_dict_24(self):
        "Dict to be compared with Eelbrain 0.24 cache"
        out = {k: v for k, v in self.as_dict().items() if v is not None}
        if isinstance(self, (SecondaryEpoch, SuperEpoch)):
            out['_rej_file_epochs'] = self.rej_file_epochs
        if isinstance(self, PrimaryEpoch) and self.n_cases:
            out['n_cases'] = self.n_cases
        if out['trigger_shift'] == 0:
            del out['trigger_shift']
        return out


class PrimaryEpoch(Epoch):
    """Epoch based on selecting events from raw file

    Attributes
    ----------
    session : str
        Session of the raw file.
    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sel',)

    def __init__(self, name, session, sel=None, n_cases=None, **kwargs):
        Epoch.__init__(self, name, **kwargs)
        self.session = session
        self.sel = typed_arg(sel, str)
        self.n_cases = typed_arg(n_cases, int)
        self.rej_file_epochs = (name,)
        self.sessions = (session,)


class SecondaryEpoch(Epoch):
    """Epoch inheriting event selection from another epoch

    sel, vars and trigger shift will be applied from the sel_epoch

    Attributes
    ----------
    sel_epoch : str
        Name of the epoch form which selection is inherited
    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sel_epoch', 'sel')
    INHERITED_PARAMS = ('tmin', 'tmax', 'decim', 'baseline',
                        'post_baseline_trigger_shift',
                        'post_baseline_trigger_shift_min',
                        'post_baseline_trigger_shift_max')

    def __init__(self, name, base, sel=None, **kwargs):
        for param in self.INHERITED_PARAMS:
            if param not in kwargs:
                kwargs[param] = getattr(base, param)

        Epoch.__init__(self, name, **kwargs)
        self.sel_epoch = base.name
        self.sel = typed_arg(sel, str)
        self.rej_file_epochs = base.rej_file_epochs
        self.session = base.session
        self.sessions = base.sessions


class SuperEpoch(Epoch):
    """Epoch combining several other epochs

    Attributes
    ----------
    sub_epochs : tuple of str
        Names of the epochs that are combined.
    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sub_epochs',)
    INHERITED_PARAMS = ('tmin', 'tmax', 'decim', 'baseline')

    def __init__(self, name, sub_epochs, **kwargs):
        for e in sub_epochs:
            if isinstance(e, SuperEpoch):
                raise TypeError("SuperEpochs can not be defined recursively")
            elif not isinstance(e, Epoch):
                raise TypeError("sub_epochs must be Epochs, got %s" % repr(e))

        if any(e.post_baseline_trigger_shift is not None for e in sub_epochs):
            err = ("Epoch definition %s: Super-epochs are merged on the level "
                   "of events and can can not contain epochs with "
                   "post_baseline_trigger_shift" % name)
            raise NotImplementedError(err)

        for param in self.INHERITED_PARAMS:
            if param in kwargs:
                continue
            values = {getattr(e, param) for e in sub_epochs}
            if len(values) > 1:
                param_repr = ', '.join(repr(v) for v in values)
                raise ValueError("All sub_epochs of a super-epoch must have "
                                 "the same setting for %r; super-epoch %r got "
                                 "{%s}." % (param, name, param_repr))
            kwargs[param] = values.pop()

        # sessions with preserved order
        sessions = []
        for e in sub_epochs:
            if e.session not in sessions:
                sessions.append(e.session)

        Epoch.__init__(self, name, **kwargs)

        self.sessions = sessions
        self.sub_epochs = tuple(e.name for e in sub_epochs)
        self.rej_file_epochs = sum((e.rej_file_epochs for e in sub_epochs), ())


class EpochCollection(Definition):
    DICT_ATTRS = ('collect',)

    def __init__(self, name, collect):
        self.name = name
        self.collect = tuple(e.name for e in collect)
        # cache dependencies
        self.sessions = {e.session for e in collect}
        self.rej_file_epochs = sum((e.rej_file_epochs for e in collect), ())
        # inherited
        for attr in SuperEpoch.INHERITED_PARAMS:
            values = {getattr(e, attr) for e in collect}
            if len(values) > 1:
                raise DefinitionError("EpochCollection base with incompatible "
                                      "values for %s: %s" % (attr, values))
            setattr(self, attr, values.pop())
