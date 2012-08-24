from eelbrain import ui
import time

p = ui.progress_monitor(i_max=10, title='Progress', message='msg 1!', cancel=True)
for i in xrange(10):
    p.message("sleeping for %s" % i)
    time.sleep(.3)
    p.advance("%s done!" % i)
    time.sleep(.2)
    """

    To remove the progress_monitor prematurely, uncomment the following three 
    lines:
    """
##    if i == 7:
##        p.terminate()
##        break

    """

    To cause a crash while the progress_monitor is shown, uncomment the 
    following two lines. To clear up the progress_monitor after the crash,
    call::

        >>> ui.kill_progress_monitors()

    """
##    if i == 6:
##        print 6/0


