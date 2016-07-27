from __future__ import division, print_function
from math import floor

import numpy as np
from scipy.stats import spearmanr


def boosting(x, y, trf_length, delta, mindelta=None, maxiter=10000):
    if mindelta is None:
        mindelta = delta
    hs = []
    for i in xrange(40):
        h, test_sse_history = boost_1seg(x, y, trf_length, delta, maxiter, i, mindelta)
        if not np.any(h):
            hs.append(h)
    h = np.mean(hs, 0)
    corr = corr_for_kernel(y, x, h, False)
    return h, corr


def boost_1seg(x, y, trf_length, delta, maxiter, segno, mindelta):
    """Basic port of svdboostV4pred

    Parameters
    ----------
    x : array (n_stims, n_times)
        Stimulus.
    y : array (n_times,)
        Dependent signal, time series to predict.
    trf_length : int
        Length of the TRF (in time samples).
    delta : scalar
        Step of the adjustment.
    maxiter : int
        Maximum number of iterations.
    segno : int [0, 39]
        which segment to use for testing
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.

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
    n_stims, n_times = x.shape
    assert y.shape == (n_times,)

    h = np.zeros((n_stims, trf_length))

    # separate training and testing signal
    test_seg_len = int(floor(x.shape[1] / 40))
    testing_range = np.arange(test_seg_len, dtype=int) + test_seg_len * segno
    training_range = np.setdiff1d(np.arange(x.shape[1], dtype=int), testing_range)
    x_test = x[:, testing_range]
    y_test = y[testing_range]
    x = x[:, training_range]
    y = y[training_range]

    # buffers
    ypred_now = np.empty(y.shape)
    ypred_next_step = np.empty(y.shape)
    ypred_test = np.empty(y_test.shape)
    y_test_error = np.empty(y_test.shape)
    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)
    y_delta = np.empty(y.shape)

    # history lists
    history = []
    test_sse_history = []
    for i_boost in xrange(maxiter):
        history.append(h.copy())

        # evaluate current h
        if np.any(h):
            # predict
            apply_kernel(x, h, ypred_now)
            apply_kernel(x_test, h, ypred_test)

            # Compute predictive power on testing data
            np.subtract(y_test, ypred_test, y_test_error)
            test_sse_history.append(np.dot(y_test_error, y_test_error[:, None])[0])
        else:
            ypred_now.fill(0)
            test_sse_history.append(np.dot(y_test, y_test[:, None])[0])

        # stop the iteration if all the following requirements are met
        # 1. more than 10 iterations are done
        # 2. The testing error in the latest iteration is higher than that in
        #    the previous two iterations
        if (i_boost > 10 and test_sse_history[-1] > test_sse_history[-2] and
                test_sse_history[-1] > test_sse_history[-3]):
            reason = "SSE(test) not improving in 2 steps"
            break

        # generate possible movements
        new_sign.fill(0)
        for ind1 in xrange(h.shape[0]):
            for ind2 in xrange(h.shape[1]):
                # y_delta = change in y from delta change in h
                y_delta[:ind2] = 0.
                y_delta[ind2:] = x[ind1, :-ind2 or None]
                y_delta *= delta

                # ypred = ypred_now + y_delta
                # error = SS(y - ypred)
                np.add(ypred_now, y_delta, ypred_next_step)
                np.subtract(y, ypred_next_step, ypred_next_step)
                e1 = np.dot(ypred_next_step, ypred_next_step[:, None])

                # ypred = y_pred_now - y_delta
                # error = SS(y - ypred)
                np.subtract(ypred_now, y_delta, ypred_next_step)
                np.subtract(y, ypred_next_step, ypred_next_step)
                e2 = np.dot(ypred_next_step, ypred_next_step[:, None])

                if e1 > e2:
                    new_error[ind1, ind2] = e2
                    new_sign[ind1, ind2] = -1
                else:
                    new_error[ind1, ind2] = e1
                    new_sign[ind1, ind2] = 1

        # If no improvements can be found reduce delta
        if new_error.min() > np.sum((y - ypred_now) ** 2):
            if delta < mindelta:
                reason = ("No improvement possible for training data, "
                          "stopping...")
                break
            else:
                delta *= 0.5
                print("No improvement, new delta=%s..." % delta)
                continue

        # update h with best movement
        bestfil = np.unravel_index(np.argmin(new_error), h.shape)
        h[bestfil] += new_sign[bestfil] * delta

        # abort if we're moving in circles
        if len(history) >= 2 and np.array_equal(h, history[-2]):
            reason = "Same h after 2 iterations"
            break
        elif len(history) >= 3 and np.array_equal(h, history[-3]):
            reason = "Same h after 3 iterations"
            break
    else:
        reason = "maxiter exceeded"

    best_iter = np.argmin(test_sse_history)
    print(reason + ' (%i iterations)' % len(test_sse_history))

    # Keep predictive power as the correlation for the best iteration
    return history[best_iter], test_sse_history


def apply_kernel(x, h, out=None):
    """Predict ``y`` by applying kernel ``h`` to ``x``"""
    if out is None:
        out = np.zeros(x.shape[1])
    else:
        out.fill(0)

    for ind in xrange(len(h)):
        out += np.convolve(h[ind], x[ind])[:len(out)]

    return out


def corr_for_kernel(y, x, h, skip_beginning=True, out=None):
    """Correlation of ``y`` and the prediction with kernel ``h``"""
    y_pred = apply_kernel(x, h)
    if skip_beginning:
        i0 = h.shape[1] - 1
        y = y[i0:]
        y_pred = y_pred[i0:]

    if out is None:
        return np.corrcoef(y, y_pred)[0, 1]
    elif out == 'rank':
        return spearmanr(y, y_pred)[0]
    elif out == 'both':
        return np.corrcoef(y, y_pred)[0, 1], spearmanr(y, y_pred)[0]
    else:
        raise ValueError("out=%s" % repr(out))
