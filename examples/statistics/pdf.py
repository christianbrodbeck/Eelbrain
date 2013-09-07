"""
Performs ANOVA and copies the ANOVA table to the clip-board. Requires the
'tex' Python module to be installed.

"""

import numpy as np
from eelbrain.eellab import *
from eelbrain.fmtxt import copy_pdf

Y = np.array([ 7, 3, 6, 6, 5, 8, 6, 7,
               7, 11, 9, 11, 10, 10, 11, 11,
               8, 14, 10, 11, 12, 10, 11, 12,
              16, 7, 11, 9, 10, 11, 8, 8,
              16, 10, 13, 10, 10, 14, 11, 12,
              24, 29, 10, 22, 25, 28, 22, 24])

A = Factor([1, 0], rep=3 * 8, name='A')
B = Factor(range(3), tile=2, rep=8, name='B')

# Independent Measures:
subject = Factor(range(8 * 6), name='subject', random=True)
anova = test.anova(Y, A * B + subject(A % B), title="Independent Measures:")
anova_table = anova.table()

# Copy pdf to clipboard:
copy_pdf(anova_table)

# The table could also be saved as pdf using eelbrain.fmtxt.save_pdf()
