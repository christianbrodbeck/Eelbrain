# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections import OrderedDict

from .._exceptions import DefinitionError
from .definitions import Definition, typed_arg


def assemble_epochs(epoch_def, epoch_default):
    epochs = {}
    secondary_epochs = OrderedDict()
    super_epochs = {}
    collections = {}
    for name, parameters in epoch_def.items():
        # into Epochs object
        if isinstance(parameters, EpochBase):
            epoch = parameters
        elif isinstance(parameters, dict):
            if 'base' in parameters:
                epoch = SecondaryEpoch(**parameters)
            elif 'sub_epochs' in parameters:
                epoch = SuperEpoch(**parameters)
            elif 'collect' in parameters:
                epoch = EpochCollection(**parameters)
            else:
                kwargs = {**epoch_default, **parameters}
                epoch = PrimaryEpoch(**kwargs)
        else:
            raise TypeError(f"Epoch {name}: {parameters!r}")

        epoch_type = type(epoch)
        if epoch_type is PrimaryEpoch:
            epoch._link(name, epochs)
            epochs[name] = epoch
        elif epoch_type is SecondaryEpoch:
            secondary_epochs[name] = epoch
        elif epoch_type is SuperEpoch:
            super_epochs[name] = epoch
        elif epoch_type is EpochCollection:
            collections[name] = epoch
        else:
            raise RuntimeError(f"epoch_type={epoch_type!r}")

    secondary_epochs.update(super_epochs)
    secondary_epochs.update(collections)
    # integrate secondary epochs (epochs with base parameter)
    name = None
    while secondary_epochs:
        if name is not None:
            del secondary_epochs[name]
        for name in secondary_epochs:
            epoch = secondary_epochs[name]
            if epoch._can_link(epochs):
                epoch._link(name, epochs)
                epochs[name] = epoch
                break
        else:
            DefinitionError(f"Can't resolve epoch dependencies for {', '.join(secondary_epochs)}")

    return epochs


class EpochBase(Definition):

    def _link(self, name, epochs):
        self.name = name


class Epoch(EpochBase):
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

    def __init__(self, tmin=-0.1, tmax=0.6, decim=5, baseline=None,
                 vars=None, trigger_shift=0., post_baseline_trigger_shift=None,
                 post_baseline_trigger_shift_min=None,
                 post_baseline_trigger_shift_max=None):
        if (post_baseline_trigger_shift is not None and
                (post_baseline_trigger_shift_min is None or
                 post_baseline_trigger_shift_max is None)):
            raise ValueError(f"{self.__class__.__name__} contains post_baseline_trigger_shift but is missing post_baseline_trigger_shift_min and/or post_baseline_trigger_shift_max")

        if baseline is None:
            if tmin >= 0:
                baseline = False
            else:
                baseline = (None, 0)
        elif baseline is False:
            pass
        elif len(baseline) != 2:
            raise ValueError(f"baseline={baseline!r}: needs to be length 2 tuple")
        else:
            baseline = (typed_arg(baseline[0], float), typed_arg(baseline[1], float))

        if not isinstance(trigger_shift, (float, str)):
            raise TypeError(f"trigger_shift={trigger_shift!r}: needs to be float or str")

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

    def __init__(self, session, sel=None, n_cases=None, **kwargs):
        Epoch.__init__(self, **kwargs)
        self.session = session
        self.sel = typed_arg(sel, str)
        self.n_cases = typed_arg(n_cases, int)
        self.sessions = (session,)

    def _link(self, name, epochs):
        Epoch._link(self, name, epochs)
        self.rej_file_epochs = (name,)


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

    def __init__(self, base, sel=None, **kwargs):
        self.sel_epoch = base
        self.sel = typed_arg(sel, str)
        self._kwargs = kwargs

    def _can_link(self, epochs):
        return self.sel_epoch in epochs

    def _link(self, name, epochs):
        base = epochs[self.sel_epoch]
        if not isinstance(base, (PrimaryEpoch, SecondaryEpoch)):
            raise DefinitionError(f"Epoch {name}, base={self.sel_epoch!r}: is {base.__class__.__name__}, needs to be PrimaryEpoch or SecondaryEpoch")
        kwargs = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            if param not in kwargs:
                kwargs[param] = getattr(base, param)
        Epoch.__init__(self, **kwargs)
        Epoch._link(self, name, epochs)
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

    def __init__(self, sub_epochs, **kwargs):
        self.sub_epochs = tuple(sub_epochs)
        self._kwargs = kwargs

    def _can_link(self, epochs):
        return all(name in epochs for name in self.sub_epochs)

    def _link(self, name, epochs):
        sub_epochs = [epochs[e] for e in self.sub_epochs]
        # check sub-epochs
        for e in sub_epochs:
            if isinstance(e, SuperEpoch):
                raise DefinitionError(f"Epoch {name}: SuperEpochs can not be defined recursively")
            elif not isinstance(e, Epoch):
                raise DefinitionError(f"Epoch {name}: sub-epochs must all by PrimaryEpochs")
            elif e.post_baseline_trigger_shift is not None:
                raise DefinitionError(f"Epoch {name}: Super-epochs are merged on the level of events and can't contain epochs with post_baseline_trigger_shift")
        # find inherited epoch parameters
        kwargs = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            if param in kwargs:
                continue
            values = {getattr(e, param) for e in sub_epochs}
            if len(values) > 1:
                param_repr = ', '.join(repr(v) for v in values)
                raise DefinitionError(f"Epoch {name}: All sub_epochs must have the same setting for {param}, got {param_repr}")
            kwargs[param] = values.pop()
        Epoch.__init__(self, **kwargs)
        Epoch._link(self, name, epochs)
        # sessions, with preserved order
        self.sessions = []
        self.rej_file_epochs = []
        for e in sub_epochs:
            if e.session not in self.sessions:
                self.sessions.append(e.session)
            self.rej_file_epochs.extend(e.rej_file_epochs)


class EpochCollection(EpochBase):
    """A collection of epochs that are loaded separately.

    For TRFs, a separate TRF will be estimated for each collected epoch (as
    opposed to a SuperEpoch, for which sub-epochs will be merged before estimating
    a single TRF).

    Parameters
    ----------
    collect : Sequence of str
        Epochs to collect.
    """
    # IMPLEMENTATION ALTERNATIVE?
    # ---------------------------
    # In analogy to standard epochs, the "model" parameter could be used to fit
    # a separate TRF per cell.
    #
    #  - Logistic complication: I would want to be able to fit only cell 1
    #    first, and later fit cell 2, without redundant refitting.
    DICT_ATTRS = ('collect',)

    def __init__(self, collect):
        self.collect = collect
        EpochBase.__init__(self)

    def _can_link(self, epochs):
        return all(name in epochs for name in self.collect)

    def _link(self, name, epochs):
        EpochBase._link(self, name, epochs)
        sub_epochs = [epochs[e] for e in self.collect]
        # sessions, with preserved order
        self.sessions = []
        self.rej_file_epochs = []
        for e in sub_epochs:
            if e.session not in self.sessions:
                self.sessions.append(e.session)
            self.rej_file_epochs.extend(e.rej_file_epochs)
