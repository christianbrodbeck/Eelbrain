# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from inspect import getargspec

from .._exceptions import DefinitionError
from .._utils.parse import find_variables


def assert_dict_has_args(d, cls, kind, name, n_internal=0):
    "Make sure the dictionary ``d`` has all keys required by ``cls``"
    argspec = getargspec(cls.__init__)
    required = argspec.args[1 + n_internal: -len(argspec.defaults)]
    missing = set(required).difference(d)
    if missing:
        raise DefinitionError(
            "%s definition %s is missing the following parameters: %s" %
            (kind, name, ', '.join(missing)))


def dict_change(old, new):
    "Readable representation of dict change"
    lines = []
    keys = set(new)
    keys.update(old)
    for key in sorted(keys):
        if key not in new:
            lines.append("%r: %r -> key removed" % (key, old[key]))
        elif key not in old:
            lines.append("%r: new key -> %r" % (key, new[key]))
        elif new[key] != old[key]:
            lines.append("%r: %r -> %r" % (key, old[key], new[key]))
    return lines


def log_dict_change(log, kind, name, old, new):
    log.warn("  %s %s changed:", kind, name)
    for line in dict_change(old, new):
        log.warn("    %s", line)


def log_list_change(log, kind, name, old, new):
    log.warn("  %s %s changed:", kind, name)
    removed = tuple(v for v in old if v not in new)
    if removed:
        log.warn("    Members removed: %s", ', '.join(map(str, removed)))
    added = tuple(v for v in new if v not in old)
    if added:
        log.warn("    Members added: %s", ', '.join(map(str, added)))


def find_epoch_vars(params):
    "Find variables used in a primary epoch definition"
    out = ()
    if params.get('sel'):
        out += find_variables(params['sel'])
    if 'trigger_shift' in params and isinstance(params['trigger_shift'], basestring):
        out += (params['trigger_shift'],)
    if 'post_baseline_trigger_shift' in params:
        out += (params['post_baseline_trigger_shift'],)
    return out


def find_epochs_vars(epochs):
    "Find variables used in all epochs"
    todo = list(epochs)
    out = {}
    while todo:
        for e in tuple(todo):
            p = epochs[e]
            if 'sel_epoch' in p:
                if p['sel_epoch'] in out:
                    out[e] = out[p['sel_epoch']] + find_epoch_vars(p)
                    todo.remove(e)
            elif 'sub_epochs' in p:
                if all(se in out for se in p['sub_epochs']):
                    out[e] = sum((out[se] for se in p['sub_epochs']),
                                 find_epoch_vars(p))
                    todo.remove(e)
            else:
                out[e] = find_epoch_vars(p)
                todo.remove(e)
    return out


def find_dependent_epochs(epoch, epochs):
    "Find all epochs whise definition depends on epoch"
    todo = set(epochs).difference(epoch)
    out = [epoch]
    while todo:
        last_len = len(todo)
        for e in tuple(todo):
            p = epochs[e]
            if 'sel_epoch' in p:
                if p['sel_epoch'] in out:
                    out.append(e)
                    todo.remove(e)
            elif 'sub_epochs' in p:
                if any(se in out for se in p['sub_epochs']):
                    out.append(e)
                    todo.remove(e)
            else:
                todo.remove(e)
        if len(todo) == last_len:
            break
    return out[1:]


def find_test_vars(params):
    "Find variables used in a test definition"
    if 'model' in params and params['model'] is not None:
        vs = set(find_variables(params['model']))
    else:
        vs = set()

    if params['kind'] == 'two-stage':
        vs.update(find_variables(params['stage_1']))

    vardef = params.get('vars', None)
    if vardef is not None:
        if isinstance(vardef, dict):
            vardef = vardef.iteritems()
        elif isinstance(vardef, tuple):
            vardef = (map(str.strip, v.split('=', 1)) for v in vardef)
        else:
            raise TypeError("vardef=%r" % (vardef,))

        for name, definition in vardef:
            if name in vs:
                vs.remove(name)
                if isinstance(definition, tuple):
                    definition = definition[0]
                vs.update(find_variables(definition))
    return vs


def typed_arg(arg, type_):
    return None if arg is None else type_(arg)


find_test_vars.__test__ = False
