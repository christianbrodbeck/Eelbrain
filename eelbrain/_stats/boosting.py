from __future__ import division, print_function
from math import floor

import numpy as np
from scipy.stats import spearmanr


def boosting(x, y, h, delta, maxiter, segno):
    """Basic port of svdboostV4pred

    Parameters
    ----------
    x : array (n_stims, n_times)
        Stimulus.
    y : array (n_times,)
        Dependent signal, time series to predict.
    h : array, (n_stims, n_rf_times)
        Initial response function.
    delta : scalar
        Step of the adjustment.
    maxiter : int
        Maximum number of iterations.
    segno : int [0, 39]
        which segment to use for testing

    Returns
    -------
    history[best_iter] : array like h
        Winning kernel.
    test_corr[best_iter] : scalar
        Test data correlation for winning kernel.
    test_rcorr[best_iter] : scalar
        Test data rank correlation for winning kernel.
    test_sse_history : list of len n_iterations
        SSE for test data at each iteration
    train_corr : list of len n_iterations
        Correlation for training data at each iteration.
    """
    # separate training and testing signal
    test_seg_len = int(floor(x.shape[1] / 40))
    testing_range = np.arange(test_seg_len, dtype=int) + test_seg_len * segno
    training_range = np.setdiff1d(np.arange(x.shape[1], dtype=int), testing_range)
    x_test = x[:, testing_range]
    y_test = y[testing_range]
    x = x[:, training_range]
    y = y[training_range]

    # history lists
    history = []
    train_corr = []
    test_corr = []
    test_rcorr = []
    test_sse_history = []
    train_sse_history = []
    for i_boost in xrange(maxiter):
        history.append(h.copy())

        # evaluate current h
        ypred_now = y * 0
        if np.any(h):
            # predict
            ypred_test = y_test * 0
            for ind in xrange(len(h)):
                ypred_now += np.convolve(h[ind], x[ind])[:len(ypred_now)]
                ypred_test += np.convolve(h[ind], x_test[ind])[:len(ypred_test)]

            # Compute predictive power: Training
            rg = slice(h.shape[1] - 1, None)
            train_corr.append(np.corrcoef(y[rg], ypred_now[rg])[0, 1])

            # Compute predictive power: Testing
            rg = slice(h.shape[1] - 1, None)
            test_corr.append(np.corrcoef(y_test[rg], ypred_test[rg])[0, 1])
            test_rcorr.append(spearmanr(y_test[rg], ypred_test[rg])[0])

            test_sse_history.append(np.sum((y_test - ypred_test) ** 2))
            train_sse_history.append(np.sum((y - ypred_now) ** 2))
        else:
            train_corr.append(np.NaN)
            test_corr.append(np.NaN)
            test_rcorr.append(np.NaN)
            test_sse_history.append(np.sum(y_test ** 2))
            train_sse_history.append(np.sum(y ** 2))
    
        # stop the iteration if all the following requirements are met
        # 1. more than 10 iterations are done
        # 2. The testing error in the latest iteration is higher than that in
        #    the previous two iterations
        if (i_boost > 10 and test_sse_history[-1] > test_sse_history[-2] and
                test_sse_history[-1] > test_sse_history[-3]):
            break

        # generate possible movements
        new_error = np.empty(h.shape)
        new_sign = np.zeros(h.shape, int)
        for ind1 in xrange(h.shape[0]):
            for ind2 in xrange(h.shape[1]):
                x_delta = delta * np.hstack((np.zeros(ind2), x[ind1, :-ind2 or None]))

                ypred = ypred_now + x_delta
                e1 = np.sum((y - ypred) ** 2)
    
                ypred = ypred_now - x_delta
                e2 = np.sum((y - ypred) ** 2)
    
                if e1 > e2:
                    new_error[ind1, ind2] = e2
                    new_sign[ind1, ind2] = -1
                else:
                    new_error[ind1, ind2] = e1
                    new_sign[ind1, ind2] = 1

        # If no improvements can be found reduce delta
        if new_error.min() > np.sum((y - ypred_now) ** 2):
            print("No BestPos")
            if delta < 0.01:
                break
            else:
                delta *= 0.5
                continue

        # update h with best movement
        bestfil = np.unravel_index(np.argmin(new_error), h.shape)
        h[bestfil] += new_sign[bestfil] * delta

        # abort if we're moving in circles
        if len(history) >= 2 and np.array_equal(h, history[-2]):
            print("Same as -2")
            break
        elif len(history) >= 3 and np.array_equal(h, history[-3]):
            print("Same as -3")
            break

    best_iter = np.argmin(test_sse_history)
    print(len(test_sse_history), 'iterations')

    # Keep predictive power as the correlation for the best iteration
    return (history[best_iter], test_corr[best_iter], test_rcorr[best_iter],
            test_sse_history, train_corr)
