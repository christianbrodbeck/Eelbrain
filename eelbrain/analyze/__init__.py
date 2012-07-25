"""
Analyze
=======

Submodules 
----------

glm 
    regression tools

table
    creating different types of tables from data
    (:py:class:`eelbrain.fmtxt.Table` objects)
test
    statistical tests for univariate data
testnd
    statistical tests for multivariate data: maps of statistical paraamaters,
    as well as cluster permutation tests


Common Kwargs
-------------
models
    match:  specify repeated measure variable
    sub:    use only a subset of the data (e.g. "sub=sex==1"). Can use list
            of values (e.g. "sub=A==[1,2]") and understand value labels for 
            factors (e.g. "sub=Shock==['strong', 'weak']) and != (e.g. 
            "sub=Shock!='no'").
plotting
    title:  Add a title to any the figure/table


combining factors/vars
----------------------
A+B combine two factors
A%B interaction effect A x B
A*B == (A + B + A%B)
A(B) A nested within B

for var objects:
cov/A one slope on cov per category in A 



example
-------
import psystat as S

Y = np.array([ 7, 3, 6, 6, 5, 8, 6, 7,
               7,11, 9,11,10,10,11,11,
               8,14,10,11,12,10,11,12])
               
A = S.factor( np.array([ 1, 2, 3]).repeat(8), name='A' )
subject = S.factor( range(8)*3, name='Subject', random=True )

print S.anova(Y, A*subject)
S.boxplot(Y, A, match=subject)


ANOVAs
------

Factors are initialised with array-like values, and can then be used to 
constructuct anova models. Use "print model" to see the effect coding of the 
model.


Classes
-------
factor: categorial factor
var:    variable (e.g. as covariate)
lm:    linear model for one response variable and several explanatory variables
    .anova()
    .reg()

Functions
---------
Analysis 1:   oneway(Y, X, ...)
              anova(Y, X, ...)
              ancova(Y, X, ...)
              pairwise(Y, X, ...)
         2:   compare(Y, first_model, test_effect)
              comparelm(model1, model2)
Data Display: astable()
Plotting:     boxplot()
              barplot()
              regplot()



Implementation
==============

Data Containers: factor, var, nonbasic_effect
---------------------------------------------
factor objects hold categorial data (one category value for each measurement) 
and var objects hold variable data. Interaction effects create nonbasic_effect
objects. These three classes all provide methods used by other classes/functions:
 .as_effects returns effect coding
 
Models
------
Any combination of data containers returns a model object. Provides .full property
returning the full model. An intercept is automatically added if none is contained
in the model (This property is used by the lm class). The .as_effects property
returns the model without intercept.

Linear Model
------------
An lm object is created with a dependent variable and an explanatory model (con-
taining at least one explanatory factor or variable).




(Written by Christian Brodbeck 2009--2011; christianmbrodbeck@gmail.com)
"""
