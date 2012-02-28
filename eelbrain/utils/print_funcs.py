'''
Some functions which print data in a different way.


Created on Feb 28, 2012

@author: christian
'''


def printdict(dictionary):
    """
    Prints only one key-value pair per line, hopefully a more readable
    representation for complex dictionaries. 
    
    TODO: multiline-values
    
    """
    items = []
    k_len = 0
    for k, v in dictionary.iteritems():
        k = repr(k)
        v = repr(v)
        k_len = max(k_len, len(k))
        items.append((k, v))
    if k_len > 75:
        raise ValueError("Key representation exceeds max len")
    v_len = 78 - k_len
    empty_k = k_len*' ' + '  '
    lines = []
    for k, v in items:
        lines.append(': '.join((k.rjust(k_len), v[:v_len])))
        if len(v) >= v_len:
            for i in xrange(v_len, len(v)+v_len-1, v_len):
                lines.append(empty_k + v[i : i+v_len])
    print '\n'.join(lines)


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
