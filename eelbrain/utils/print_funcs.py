'''
Some functions which print data in a different way.


Created on Feb 28, 2012

@author: christian
'''
import os
import numpy as np


def dicttree(dictionary):
    """readable repr for a hierarchical dictionary"""
    print os.linesep.join(_dict_repr(dictionary))

def _dict_repr(dictionary, indent=0):
    out = []
    head = ' ' * indent
    for k, v in dictionary.iteritems():
        if isinstance(v, dict):
            out.append(head + repr(k))
            out.extend(_dict_repr(v, indent=indent + 3))
        else:
            r = repr(v)
            if len(r) > 40:
                if isinstance(v, np.ndarray):
                    r = "<array, shape = %s>" % repr(v.shape)
                else:
                    r = str(type(v))
            out.append(head + repr(k) + ': ' + r)
    return out



def printdict(dictionary, w=100, fmt='%r', sort=True):
    """
    Prints only one key-value pair per line, hopefully a more readable
    representation for complex dictionaries. 
    
    sort : bool
        Sort keys


    TODO: multiline-values
    
    """
    print strdict(dictionary, w=w, fmt=fmt, sort=sort)


def strdict(dictionary, w=100, fmt='%r', sort=True):
    items = []
    k_len = 0

    if sort:
        keys = sorted(dictionary)
    else:
        keys = dictionary.keys()

    for k in keys:
        v = dictionary[k]
        k = str(k) if isinstance(k, tuple) else fmt % k
        v = str(v) if isinstance(v, tuple) else fmt % v
        k_len = max(k_len, len(k))
        items.append((k, v))
    if k_len > w - 5:
        raise ValueError("Key representation exceeds max len")
    v_len = w - 2 - k_len
    empty_k = k_len*' ' + '  '
    lines = []
    for k, v in items:
        lines.append(': '.join((k.ljust(k_len), v[:v_len])))
        if len(v) >= v_len:
            for i in xrange(v_len, len(v)+v_len-1, v_len):
                lines.append(empty_k + v[i : i+v_len])
    lines[0] = '{' + lines[0]
    lines[-1] = lines[-1] + '}'
    return ',\n '.join(lines)


def printlist(list_obj, rep=True):
    """
    print each element of a list on a new line
    
    :arg rep: print repr representation (vs str representation)
    
    """
    for line in list_obj:
        if rep:
            print repr(line)
        else:
            print str(line)
