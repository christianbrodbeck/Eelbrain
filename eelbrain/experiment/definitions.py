# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from .._utils.parse import find_variables


def find_test_vars(params):
    "Find variables used in a test definition"
    if params['kind'] == 'two-stage':
        vs = set(find_variables(params['stage 1']))
        if 'vars' in params:
            for name, definition in params['vars'].iteritems():
                vs.remove(name)
                if isinstance(definition, tuple):
                    definition = definition[0]
                vs.update(find_variables(definition[1]))
    else:
        vs = find_variables(params['model'])
    return vs
