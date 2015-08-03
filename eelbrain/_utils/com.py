'''Communication utilities'''
from email.mime.text import MIMEText
import keyring
import pdb
import smtplib
import socket
import traceback

from .basic import logger
from .system import caffeine
from . import ui


NOOB_DOMAIN = "Eelbrain"
NOOB_ADDRESS = 'n00b.eelbrain@gmail.com'


def get_smtpserver(password, new_password=False):
    smtpserver = smtplib.SMTP('smtp.gmail.com', 587)
    smtpserver.ehlo()
    smtpserver.starttls()
    while True:
        try:
            smtpserver.login(NOOB_ADDRESS, password)
            if new_password:
                keyring.set_password(NOOB_DOMAIN, NOOB_ADDRESS, password)
            return smtpserver
        except smtplib.SMTPAuthenticationError:
            password = ui.ask_str("Eelbrain notifier password invalid. Please "
                                  "enter valid password.", "Notifier Password")
            if password:
                new_password = True
            else:
                raise


def send_email(to, subject, body, password):
    """Send an email notification"""
    msg = MIMEText(body)
    msg['Subject'] = subject
    host = socket.gethostname()
    msg['From'] = 'Eelbrain on %s <%s>' % (host, NOOB_ADDRESS)
    msg['To'] = to

    smtpserver = get_smtpserver(password)
    smtpserver.sendmail(NOOB_ADDRESS, to, msg.as_string())
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
    def __init__(self, to, name='job', crash_info_func=None, debug=True):
        """
        Parameters
        ----------
        to : str
            Email address of the recipient.
        name : str
            Name of the job (will be included in subject line).
        crash_info_func : None | callable
            Will be called upon crash to produce a string that will be included
            in the crash report.
        debug : bool
            If the task crashes, start pdb instead of exiting.
        """
        # get the password
        password = keyring.get_password(NOOB_DOMAIN, NOOB_ADDRESS)
        if password is None:
            password = ui.ask_str("Please enter the Eelbrain notifier "
                                  "password.", "Notifier Password")
            # test it
            print "Validating password..."
            smtpserver = get_smtpserver(password, True)
            smtpserver.close()

        self.to = to
        self.name = name
        self.crash_info_func = crash_info_func
        self.debug = debug
        self._password = password

    def __enter__(self):
        logger.info("Notification enabled...")
        caffeine.__enter__()
        return self

    def __exit__(self, type_, value, traceback_):
        host = socket.gethostname()
        caffeine.__exit__(type_, value, traceback_)
        if isinstance(value, Exception):
            error = type_.__name__
            temp = '{host} encountered {error}: {value} in {task}'
            event = temp.format(host=host, error=error, value=value,
                                task=self.name)
            info = []

            # traceback
            tb_items = traceback.format_tb(traceback_)
            error_message = "%s: %s\n" % (error, value)
            tb_items.append(error_message)
            tb_str = '\n'.join(tb_items)
            info.append(tb_str)

            # object info
            if self.crash_info_func:
                info.extend(self.crash_info_func())

            self.send(event, info)

            # drop into pdb
            if self.debug:
                traceback.print_exc()
                print ''
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
        body = '\n\n\n'.join(map(unicode, info))
        send_email(self.to, subject, body, self._password)
