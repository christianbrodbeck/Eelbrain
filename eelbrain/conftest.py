# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Pytest configuration

Skipping slow tests: https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--no-gui", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "framework_build: requires framework build")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if config.getoption("--no-gui"):
        skip = pytest.mark.skip(reason="requires framework build")
        for item in items:
            if "framework_build" in item.keywords:
                item.add_marker(skip)
