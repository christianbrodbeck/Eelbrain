'''Communication utilities'''
import bz2
from email.mime.text import MIMEText
import os
import smtplib


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
    def __init__(self, to, name='job'):
        """
        Parameters
        ----------
        to : str
            Email address of the recipient.
        name : str
            Name of the job (will be included in subject line).
        """
        if not os.path.exists(_pwd_fname):
            err = "File required for notification not found: %r" % _pwd_fname
            raise IOError(err)

        self.to = to
        self.name = name

    def __enter__(self):
        self.msg = []
        return self

    def add(self, note):
        "Add a note to the notification"
        self.msg.append(unicode(note))

    def __exit__(self, type_, value, traceback):
        body = "\n\n".join(map(unicode, self.msg))
        if isinstance(value, Exception):
            result = 'failed'
            body = '\n\n'.join(map(unicode, (body, type_, value)))
        else:
            result = 'finished'
        subject = '%s %s' % (self.name, result)
        send_email(self.to, subject, body)

