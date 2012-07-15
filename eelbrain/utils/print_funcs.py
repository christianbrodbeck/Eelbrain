'''
Some functions which print data in a different way.


Created on Feb 28, 2012

@author: christian
'''


def printdict(dictionary, w=100, fmt='%r'):
    """
    Prints only one key-value pair per line, hopefully a more readable
    representation for complex dictionaries. 
    
    TODO: multiline-values
    
    """
    print strdict(dictionary, w=w, fmt=fmt)


def strdict(dictionary, w=100, fmt='%r'):
    items = []
    k_len = 0
    for k, v in dictionary.iteritems():
        k = fmt % k
        v = fmt % v
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
