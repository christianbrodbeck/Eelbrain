'''Communication utilities'''
import bz2
from email.mime.text import MIMEText
import os
import pdb
import smtplib
import socket
import traceback

from .basic import logger


_pwd_fname = os.path.expanduser('~/.eelbrain_n00b')


def send_email(to, subject, body):
    """Send an email notification"""
    gmail_user = 'n00b.eelbrain@gmail.com'
    with open(_pwd_fname) as f:
        pwd = bz2.decompress(f.read())

    msg = MIMEText(body)
    msg['Subject'] = subject
    host = socket.gethostname()
    msg['From'] = 'Eelbrain on %s <%s>' % (host, gmail_user)
    msg['To'] = to

    smtpserver = smtplib.SMTP('smtp.gmail.com', 587)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.login(gmail_user, pwd)

    smtpserver.sendmail(gmail_user, to, msg.as_string())
    smtpserver.close()


class Notifier(object):
    """
    A notification email sender supporting ``with`` statements

    Examples
    --------
    To receive a message after a task has been executed:

    >>> notifier = Notifier('me@somewhere.com')
    >>> with notifier:
    ...     do_task()
    ...

    """
    def __init__(self, to, name='job', state_func=None, debug=True):
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
        debug : bool
            If the task crashes, start pdb instead of exiting.
        """
        if not os.path.exists(_pwd_fname):
            err = "File required for notification not found: %r" % _pwd_fname
            raise IOError(err)

        self.to = to
        self.name = name
        self.state_func = state_func
        self.debug = debug

    def __enter__(self):
        logger.info("Notification enabled...")
        return self

    def __exit__(self, type_, value, traceback_):
        host = socket.gethostname()
        if isinstance(value, Exception):
            error = type_.__name__
            temp = '{host} encountered {error}: {value} in {task}'
            event = temp.format(host=host, error=error, value=value,
                                task=self.name)
            info = []

            # state description
            if self.state_func:
                state_desc = self.state_func()
                info.append(state_desc)

            # traceback
            tb_items = traceback.format_tb(traceback_)
            tb_str = '\n'.join(tb_items)
            info.append(tb_str)

            # error
            info.append("%s: %s" % (error, value))

            if self.debug:
                info.append("Starting PDB...")
            else:
                info.append("Terminating...")

            self.send(event, info)

            # drop into pdb
            if self.debug:
                traceback.print_exc()
                pdb.post_mortem(traceback_)
        else:
            event = '{host} finished {task}'.format(host=host, task=self.name)
            self.send(event)

    def send(self, subject, info=[]):
        """Send an email message

        Parameters
        ----------
        subject : str
            Email subject line.
        info : list of str
            Email body; successive entries are joined with two line breaks.
        """
        body = '\n\n'.join(map(unicode, info))
        send_email(self.to, subject, body)
