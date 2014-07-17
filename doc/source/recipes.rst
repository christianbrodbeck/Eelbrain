*******
Recipes
*******

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

    >>> ds_subject
    <Dataset 'example' n_cases=145 {'predictor':V, 'data':Vnd}>
    >>> ds_subject['data']
    <NDVar 'data': 145 (case) X 5120 (source) X 76 (time)>
    >>> print ds_subject
    predictor
    ---------
    1.9085
    0.30836
    -0.58802
    0.29686
    ...

The regression coefficient can be calculated the following way::

    >>> beta = ds_subject.eval("data.ols(predictor)")
    >>> beta
    <NDVar 'ols': 1 (case) X 5120 (source) X 76 (time)>

Thus, in order to collect beta values for each subject, you would loop through
subjects. We will call the NDVar with beta values 'beta'::

    >>> subjects = []
    >>> betas = []
    >>> for subject in ['R0001', 'R0002', 'R0003']:
    ...     ds_subject = my_load_ds_for_subject_function(subject)
    ...     beta = ds_subject.eval("data.ols(predictor, 'beta')")
    ...     subjects.append(subject)
    ...     betas.append(beta)
    ...
    >>> ds = Dataset()
    >>> ds['subject'] = Factor(subjects, random=True)
    >>> ds['beta'] = combine(betas)

Now you can perform a one-sample t-test::

    >>> res = testnd.ttest_1samp('beta', ...)

And analyze the results as for other nd-tests.
