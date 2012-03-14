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
import tempfile
import re

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


class marker_avg_file:
    def __init__(self, path):
        # Parse marker file, based on Tal's pipeline:
        regexp = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
        output_lines = []
        for line in open(path):
            match = regexp.search(line)
            if match:
                output_lines.append('\t'.join(match.groups()))
        txt = '\n'.join(output_lines)
        
        fd, self.path = tempfile.mkstemp(suffix='hpi', text=True)
        f = os.fdopen(fd, 'w')
        f.write(txt)
        f.close()
    
    def __del__(self):
        os.remove(self.path)




def kit2fiff(meg_sdir=None, ename='eref3', sfreq=250, aligntol=25, **more_kwargs):
    """
    Reads multiple input files and combines them into a fiff file that can be
    used with mne. Implemented after Gwyneth's Manual. Requires th following 
    files to be in the subject's meg-directory::
    
        data/
             <subject>_<experiment>_export.txt
        parameters/
                   <subject>_<experiment>_electrodes.elp
                   <subject>_<experiment>_headshape.hsp
                   <subject>_<experiment>_markers_average.txt
        
        + [sns.txt]
    
    
    Arguments:
    
    ename : str
        The experiment name as it appears in file names.
        
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
    fmt = (subject, ename)
    mapath = os.path.join(param_dir, '%s_%s_markers_average.txt' % fmt)
    mafile = marker_avg_file(mapath)
    
    elp_file = os.path.join(param_dir, '%s_%s_electrodes.elp' % fmt)
    hsp_file = os.path.join(param_dir, '%s_%s_headshape.hsp' % fmt)
    hpi_file = mafile.path
    sns_file = '~/Documents/Eclipse/Eelbrain\ Reloaded/aux_files/sns.txt'
    data_file = os.path.join(meg_sdir, 'data', '%s_%s_export.txt' % fmt)
    
    # out file path
    out_dir = os.path.join(meg_sdir, 'myfif')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_file = os.path.join(out_dir, '%s_raw.fif' % subject)
    if os.path.exists(out_file):
        if not ui.ask("Overwrite?", "Target File Already Exists at: %r. Should "
                      "it be replaced?" % out_file):
            return
    
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


