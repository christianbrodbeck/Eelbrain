# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Definitions that require ``pytest.config``

Relegate some functionality to this module to avoid importing ``pytest.config``,
which is undefined unless testing with ``$ pytest``, in ``eelbrain.testing``.
"""
import pytest


slow_test = pytest.mark.skipif(not pytest.config.option.slow_tests, reason="Slow test skipped without --slowtests option")
