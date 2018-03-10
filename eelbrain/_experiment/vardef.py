from .._data_obj import Var, asfactor

import numpy as np

from .definitions import DefinitionError


def as_vardef_var(v):
    "Coerce ds.eval() output for use as variable"
    if isinstance(v, np.ndarray):
        if v.dtype.kind == 'b':
            return Var(v.astype(int))
        return Var(v)
    return v


class VarDef(object):

    def apply(self, ds, e):
        raise NotImplementedError


class EvalVar(VarDef):

    def __init__(self, code):
        assert isinstance(code, str)
        self.code = code

    def __repr__(self):
        return "EvalVar(%r)" % self.code

    def __eq__(self, other):
        return isinstance(other, EvalVar) and other.code == self.code

    def apply(self, ds, e):
        return as_vardef_var(ds.eval(self.code))


class LabelVar(VarDef):

    def __init__(self, source, codes):
        self.source = source
        self.codes = codes

    def __repr__(self):
        return "LabelVar(%r, %r)" % (self.source, self.codes)

    def __eq__(self, other):
        return (isinstance(other, LabelVar) and other.source == self.source
                and other.codes == self.codes)

    def apply(self, ds, e):
        v = asfactor(self.source, ds=ds).as_var(self.codes, 0)
        return as_vardef_var(v)


class GroupVar(VarDef):

    def __init__(self, groups):
        self.groups = sorted(groups)

    def __repr__(self):
        return "GroupVar(%r)" % (self.groups,)

    def __eq__(self, other):
        return isinstance(other, GroupVar) and other.groups == self.groups

    def apply(self, ds, e):
        return e.label_groups(ds['subject'], self.groups)


def parse_named_vardef(string):
    if '=' not in string:
        raise DefinitionError("var %r: needs '='" % (string,))
    name, vdef = string.split('=', 1)
    return name.strip(), parse_vardef(vdef)


def parse_vardef(string):
    string = string.strip()
    if string.startswith('group:'):
        return GroupVar(map(str.strip, string[6:].split(',')))
    else:
        return EvalVar(string)


class Vars(object):

    def __init__(self, arg):
        if isinstance(arg, str):
            arg = (arg,)
        elif isinstance(arg, dict):
            arg = arg.items()
        elif not isinstance(arg, (tuple, list)):
            raise TypeError("vars=%s; needs to be atrs, dict or tuple" % (arg,))

        items = []
        for item in arg:
            if isinstance(item, str):
                name, vdef = parse_named_vardef(item)
            else:
                name, vdef = item
                if isinstance(vdef, str):
                    vdef = parse_vardef(vdef)
                elif isinstance(vdef, VarDef):
                    pass
                else:
                    vdef = LabelVar(*vdef)
            items.append((name, vdef))
        self.items = items

    def __repr__(self):
        return "Vars(%s)" % (self.items,)

    def __eq__(self, other):
        return isinstance(other, Vars) and other.items == self.items

    def apply(self, ds, e):
        for name, vdef in self.items:
            ds[name] = vdef.apply(ds, e)
