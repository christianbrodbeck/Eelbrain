# requires: Fox_Prestige_data.txt
#
# Results cross-checked with Fox 2008, p. 139:
# --------------------------------------------
##
##                 	   SS      	df 	  MS      	    F      	 p   
##-----------------------------------------------------------------
##income           	 1131.9002 	 1 	1131.9002 	28.3544*** 	.0000
##education        	 1067.9831 	 1 	1067.9831 	26.7532*** 	.0000
##type             	  591.1628 	 2 	 295.5814 	 7.4044**  	.0011
##income x type    	  951.7704 	 2 	 475.8852 	11.9210*** 	.0000
##education x type 	  238.3965 	 2 	 119.1982 	 2.9859    	.0556
##Residuals        	 3552.8611 	89 	  39.9198 	           	     
##-----------------------------------------------------------------
##Total            	28346.8756 	97 	          	           	     
##

from eelbrain import *


filepath = 'Fox_Prestige_data.txt'

# load data
ds = load.txt.tsv(filepath, delimiter=None)

# exclude cases with missing type
ds2 = ds[ds['type'] != 'NA']

# ANOVA
print test.anova('prestige', '(income + education) * type', ds=ds2)
