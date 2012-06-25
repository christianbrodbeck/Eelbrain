"""
Performs ANOVA and copies the ANOVA table to the clip-board. Requires the 
'tex' Python module to be installed.

"""

import numpy as np
import psystats as S


Y = np.array([[ 7, 3, 6, 6, 5, 8, 6, 7],
              [ 7,11, 9,11,10,10,11,11],
              [ 8,14,10,11,12,10,11,12],
              [16, 7,11, 9,10,11, 8, 8],
              [16,10,13,10,10,14,11,12],
              [24,29,10,22,25,28,22,24]])

A = S.factor(np.array([1,0]).repeat(3*8), name='A')
B = S.factor(np.array(range(3)*2).repeat(8), name='B')

# Independent Measures:
subject = S.factor(np.array(range(8*6)), name='subject', random=True)
anova = S.anova(Y, A*B+subject(A%B), title="Independent Measures:")
anova_table = anova.anova()

S.copy_pdf(anova_table)