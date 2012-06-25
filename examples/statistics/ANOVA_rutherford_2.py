# Rutherford (2001) Examples, cross-checked results:
#
# factorial anova 
# 
#Independent Measures (p. 53):
##               	  SS     	df 	  MS     	    F      	 p   
##_____________________________________________________________
##A              	432.0000 	 1 	432.0000 	47.7474*** 	.0000
##B              	672.0000 	 2 	336.0000 	37.1368*** 	.0000
##A x B          	     224 	 2 	     112 	12.3789*** 	.0001
##subject(A x B) 	380.0000 	42 	  9.0476 	           	     
##_____________________________________________________________
##Total          	    1708 	47 	         	           	     
#
#
#Repeated Measure (p. 86):
##                	  SS     	df 	  MS     	    F      	 p   
##_________________________________________________________________
##A               	432.0000 	 1 	432.0000 	40.1416*** 	.0004
##B               	672.0000 	 2 	336.0000 	29.2174*** 	.0000
##A x B           	224.0000 	 2 	112.0000 	17.1055*** 	.0002
##subject         	 52.0000 	 7 	  7.4286 	 0.7927    	.5984
##A x subject     	 75.3333 	 7 	 10.7619 	 1.6436    	.2029
##B x subject     	161.0000 	14 	 11.5000 	 1.7564    	.1519
##A x B x subject 	 91.6667 	14 	  6.5476 	           	     
##_________________________________________________________________
##Total           	    1708 	47 	         	           	     

import numpy as np
from eelbrain.eellab import *


Y = np.array([ 7, 3, 6, 6, 5, 8, 6, 7,
               7,11, 9,11,10,10,11,11,
               8,14,10,11,12,10,11,12,
              16, 7,11, 9,10,11, 8, 8,
              16,10,13,10,10,14,11,12,
              24,29,10,22,25,28,22,24])

A = factor([1,0], rep=3*8, name='A')
B = factor(range(3), tile=2, rep=8, name='B')

# Independent Measures:
subject = factor(range(8*6), name='subject', random=True)
print test.anova(Y, A*B+subject(A%B), title="Independent Measures:")

# Repeated Measure:
subject = factor(range(8), tile=6, name='subject', random=True)
print test.anova(Y, A * B * subject, title="Repeated Measure:")

