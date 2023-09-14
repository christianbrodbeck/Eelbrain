# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain._utils.null_os import user_activity


def test_user_activity():
    with user_activity:
        with user_activity:
            pass

    user_activity.__enter__()
    user_activity.__enter__()
    user_activity.__exit__(None, None, None)
    user_activity.__exit__(None, None, None)
