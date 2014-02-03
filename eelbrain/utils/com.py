'''Communication utilities'''
import bz2
from email.mime.text import MIMEText
import os
import smtplib
import traceback


_pwd_fname = os.path.expanduser('~/.eelbrain_n00b')


def send_email(to, subject, body):
    """Send an email notification"""
    gmail_user = 'n00b.eelbrain@gmail.com'
    with open(_pwd_fname) as f:
        pwd = bz2.decompress(f.read())

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'The Eelbrain N00b <%s>' % gmail_user
    msg['To'] = to

    smtpserver = smtplib.SMTP('smtp.gmail.com', 587)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.login(gmail_user, pwd)

    smtpserver.sendmail(gmail_user, to, msg.as_string())
    smtpserver.close()


class Notifier(object):
    """
    A notification sender supporting ``with`` statements

    Examples
    --------
    To receive a message after a task has been executed:

    >>> notifier = Notifier('me@somewhere.com')
    >>> with notification:
    ...     do_task
    ...

    """
    def __init__(self, to, name='job', state_func=None):
        """
        Parameters
        ----------
        to : str
            Email address of the recipient.
        name : str
            Name of the job (will be included in subject line).
        state_func : None | callable
            Will be called upon crash to produce a string that will be included
            in the crash report.
        """
        if not os.path.exists(_pwd_fname):
            err = "File required for notification not found: %r" % _pwd_fname
            raise IOError(err)

        self.to = to
        self.name = name
        self.state_func = state_func

    def __enter__(self):
        self.msg = []
        return self

    def add(self, note):
        "Add a note to the notification"
        self.msg.append(unicode(note))

    def __exit__(self, type_, value, traceback_):
        items = self.msg[:]
        if isinstance(value, Exception):
            result = '%s: %s' % (type_.__name__, value)

            # state description
            if self.state_func:
                state_desc = self.state_func()
                items.append(state_desc)

            # traceback
            tb_items = traceback.format_tb(traceback_)
            tb_str = '\n'.join(tb_items)
            items.append(tb_str)
            items.append(result)
        else:
            result = 'finished'
        body = '\n\n'.join(map(unicode, items))
        subject = '%s %s' % (self.name, result)
        send_email(self.to, subject, body)

