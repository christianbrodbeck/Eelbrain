'''

for permission errors: try ``os.chmod`` or ``os.chown``


subprocess documentation
------------------------

http://docs.python.org/library/subprocess.html
http://www.doughellmann.com/PyMOTW/subprocess/
http://explanatorygap.net/2010/05/10/python-subprocess-over-shell/
http://codeghar.wordpress.com/2011/12/09/introduction-to-python-subprocess-module/


Created on Mar 4, 2012

@author: christian
'''

import os
import subprocess

from eelbrain import ui


__all__ = ['kit2fiff']


_mne_dir = '~/unix_apps/mne-2.7.3'

# set environment variables for MNE
os.environ['MNE_ROOT'] = _mne_dir

_mne_bin = os.path.join(_mne_dir, 'bin')
if 'PATH' in os.environ:
    os.environ['PATH'] += ':%s' % _mne_bin
else:
    os.environ['PATH'] = _mne_bin


def kit2fiff(meg_sdir=None, sfreq=250, aligntol=25, **more_kwargs):
    """
    Reads multiple input files and combines them into a fiff file that can be
    used with mne. Implemented after Gwyneth's Manual
    
    Arguments:
    
    sfreq : scalar
        samplingrate of the data
    
    meg_sdir : path(str)
        Path to the subjects's meg directory. If ``None``, a file dialog is 
        displayed to ask for the directory.
    
    more_kwargs :
        more keyword arguments (see MNE manual p. 224) e.g., ``lowpass``
    
    """
    if meg_sdir is None:
        msg = "Select Subject's Meg Directory"
        meg_sdir = ui.ask_dir(msg, msg, True)
    
    param_dir = os.path.join(meg_sdir, 'parameters')
    assert os.path.exists(param_dir)
    subject = os.path.basename(meg_sdir)
    
    elp_file = os.path.join(param_dir, '%s_eref3_electrodes.elp' % subject)
    hsp_file = os.path.join(param_dir, '%s_eref3_headshape.hsp' % subject)
    hpi_file = os.path.join(param_dir, '%s_eref3_markers.hpi' % subject)
    sns_file = '~/Documents/Eclipse/Eelbrain\ Reloaded/aux_files/sns.txt'
    data_file = '~/Documents/Data/eref/meg/R0368/data/%s_eref3_exported.txt' %  subject
    out_file = os.path.join(meg_sdir, 'myfif', '%s_raw.fif' % subject)
    
    kwargs = {'elp': elp_file,
              'hsp': hsp_file,
              'sns': sns_file,
              'hpi': hpi_file,
              'raw': data_file,
              'out': out_file,
              'stim': '161:162:163:164:165:166:167:168',
              'sfreq': sfreq,
              'aligntol': aligntol}
    
    kwargs.update(more_kwargs)
    
    _run('mne_kit2fiff', kwargs)
#    print_funcs.printlist(cmd)


def _run(cmd, kwargs=None):
    """
    cmd:
        string command
    
    kwargs : (optional)
        dictionary of arguments that are formatted with:
        ``"--%s %s" % (key, value)"``
    
    """
    if kwargs:
        args = ' '.join('--%s %s' % item for item in kwargs.iteritems())
        cmd = ' '.join((cmd, args))
#    source ~/unix_apps/mne-2.7.3/bin/mne_setup_sh
    sp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           shell=True)
    
    stdout, stderr = sp.communicate()
    print ">COMMAND:"
    print cmd
    print ">OUT:"
    print stdout
    print '>ERROR:'
    print stderr


