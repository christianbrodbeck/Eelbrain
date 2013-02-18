'''Communication utilities'''
import bz2
from email.mime.text import MIMEText
import os
import smtplib


def send_email(to, subject, body):
    """Send an email notification"""
    gmail_user = 'n00b.eelbrain@gmail.com'
    pwd_fname = os.path.expanduser('~/.eelbrain_n00b')
    with open(pwd_fname) as f:
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
