# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import parser


FLOAT_PATTERN = "^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
POS_FLOAT_PATTERN = "^[+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
INT_PATTERN = "^-?\d+$"
# matches float as well as NaN:
FLOAT_NAN_PATTERN = "^([Nn][Aa][Nn]$)|([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$"


def find_variables(expr):
    """Find the variables participating in an expressions

    Returns
    -------
    variables : tuple of str
        Variables occurring in expr.
    """
    return _find_vars(parser.expr(expr).totuple())


def _find_vars(st):
    if isinstance(st, str):
        return ()
    elif st[0] == 318:
        if st[1][0] == 1:
            return st[1][1],
        else:
            return ()
    else:
        return sum((_find_vars(b) for b in st[1:]), ())
