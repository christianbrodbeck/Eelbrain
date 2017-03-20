# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from .._utils.parse import find_variables


class DefinitionError(Exception):
    "MneExperiment definition error"


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
    if 'model' in params:
        vs = set(find_variables(params['model']))
    else:
        vs = set()

    if params['kind'] == 'two-stage':
        vs.update(find_variables(params['stage 1']))

    if 'vars' in params:
        vardef = params['vars']
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

find_test_vars.__test__ = False
