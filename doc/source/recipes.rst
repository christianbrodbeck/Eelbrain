.. currentmodule:: eelbrain

.. testsetup:: *

    from eelbrain import *
    ds_subject = datasets.get_ols()
    my_load_ds_for_subject_function = lambda: ds_subject


*******
Recipes
*******

.. _recipe-regression:

^^^^^^^^^^^^^^^^^
Regression Design
^^^^^^^^^^^^^^^^^

The influence of a continuous predictor on single trial level can be tested by
first calculating regression coefficients for each subject, and then performing
a one sample t-test across subjects to test the null hypothesis that between
subjects, the regression coefficient does not differ significantly from 0.

Assuming that ``ds_subject`` is a :class:`Dataset` containing single trial data
for one subject, with ``data`` the dependent variable and ``pred`` the
predictor::

.. doctest::

    >>> ds_subject
    <Dataset n_cases=120 {'predictor':V, 'y':Vnd}>
    >>> ds_subject['y']
    <NDVar 'y': 120 (case) X 100 (time)>    >>> ds_subject['data']
    >>> print ds_subject[:4]
    predictor
    ---------
    1.7641   
    0.40016  
    0.97874  
    2.2409   

The regression coefficient can be calculated the following way::

    >>> beta = ds_subject.eval("y.ols(predictor)")
    >>> beta
    <NDVar '': 1 (case) X 100 (time)>

Thus, in order to collect beta values for each subject, you would loop through
subjects. We will call the NDVar with beta values 'beta'::

    >>> subjects = ['R0001', 'R0002', 'R0003']
    >>> betas = []
    >>> for subject in subjects:
    ...     ds_subject = my_load_ds_for_subject_function(subject)
    ...     beta = ds_subject.eval("y.ols(predictor, 'beta')")
    ...     betas.append(beta)
    ...
    >>> ds = Dataset()
    >>> ds['subject'] = Factor(subjects, random=True)
    >>> ds['beta'] = combine(betas)

Now you can perform a one-sample t-test::

    >>> res = testnd.ttest_1samp('beta', ds=ds)

And analyze the results as for other nd-tests.


Regression on Residuals
-----------------------

.. testsetup::

    ds_subject = datasets.get_ols(('predictor', 'confound', 'confound_2'))

In order to account for the influence of a confound variable ``confound`` on 
``data``, the regression of ``data`` on ``predictor`` can be performed on the 
residuals of the regression of ``data`` on ``confound``. For this purpose the
:class:`NDVar` has an :meth:`NDVar.residuals` method::

    >>> ds_subject['residuals'] = ds_subject.eval("y.residuals(confound)")
    >>> beta = ds_subject.eval("residuals.ols(predictor)")

The same method can be used with several confounds (all of type :class:`Var`)::

    >>> ds_subject['residuals'] = ds_subject.eval("data.residuals(confound + confound_2)")
    >>> beta = ds_subject.eval("residuals.ols(predictor)")
