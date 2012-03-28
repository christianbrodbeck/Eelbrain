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


__all__ = ['kit2fiff', 'set_mne_dir']

# keep track of whether the mne dir has been successfully set
_mne_dir = None


def set_mne_dir(path):
    """
    Set the directory where mne is installed. E.g. ::
    
        >>> set_mne_dir('~/unix_apps/mne-2.7.3')
    
    """
    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(path):
        raise IOError("%r does not exist" % path)
    
    # set environment variables for MNE
    os.environ['MNE_ROOT'] = path
    
    if 'PATH' in os.environ:
        os.environ['PATH'] += ':%s' % path
    else:
        os.environ['PATH'] = path
    
    global _mne_dir
    _mne_dir = path
    



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



def _format_path(path, fmt, is_new=False):
    "helper function to format the path to mne files"
    if not isinstance(path, basestring):
        path = os.path.join(*path)
    
    if fmt:
        path = path.format(**fmt)
    
    # test the path
    path = os.path.expanduser(path)
    if is_new or os.path.exists(path):
        return path
    else:
        raise IOError("%r does not exist" % path)
    


def kit2fiff(meg_sdir=None, experiment='eref3', sfreq=500, aligntol=25,
             mrk = ('{meg_sdir}', 'parameters', '{subject}_{experiment}_markers.txt'),
             elp = ('{meg_sdir}', 'parameters', '{subject}_{experiment}.elp'),
             hsp = ('{meg_sdir}', 'parameters', '{subject}_{experiment}.hsp'),
             sns = ('~/Documents/Eclipse/Eelbrain Reloaded/aux_files/sns.txt',),
             raw = ('{meg_sdir}', 'data', '{subject}_{experiment}_export.txt'),
             out = ('{meg_sdir}', 'myfif', '{subject}_{experiment}_raw.fif'),
             stim=xrange(168, 160, -1), lowpass=100, highpass=0, stimthresh=1):
    """
    Calls the mne2fiff mne binary which reads multiple input files and combines
    them into a fiff file. Implemented after Gwyneth's Manual; for more 
    information see the mne manual (p. 222). 
    
    All filenames can be specified either as string, or as tuple of strings.
    Tuples are converted to paths with::
    
        >>> os.path.join(*arg).format(**fmt)
    
    where ``fmt`` contains the following entries:
    
    ``meg_sdir``
        input argument
    ``experiment``
        input argument
    ``subject``
        last directory name in ``meg_sdir``
    
    Directories can contain the user home shortcut (``~``). Other Arguments are:
    
    mrk, elp, hsp, raw, out, sns : str or tuple (see above)
        input files
        
    experiment : str
        The experiment name as it appears in file names.
    
    highpass : scalar
        The highpass filter corner frequency (only for file info, does not
        filter the data). 0 Hz for DC recording.
    
    lowpass : scalar
        like highpass
    
    meg_sdir : str or tuple (see above)
        Path to the subjects's meg directory. If ``None``, a file dialog is 
        displayed to ask for the directory.
    
    sfreq : scalar
        samplingrate of the data
    
    stimthresh : scalar
        The threshold value used when synthesizing the digital trigger channel
    
    stim : iterable over ints
        trigger channels that are used to reconstruct the event cue value from 
        the separate trigger channels. The default ``xrange(168, 160, -1)`` 
        reconstructs the values sent through psychtoolbox.
    
    """
    if not _mne_dir:
        raise IOError("mne-dir not set. See mne_link.set_mne_dir.")
    
    if meg_sdir is None:
        msg = "Select Subject's Meg Directory"
        meg_sdir = ui.ask_dir(msg, msg, True)
    else:
        meg_sdir = os.path.expanduser(meg_sdir)
        
    fmt = {'subject': os.path.basename(meg_sdir),
           'experiment': experiment,
           'meg_sdir': meg_sdir,
           }
    
    # expand all the path arguments
    mrk_path = _format_path(mrk, fmt)
    elp_file = _format_path(elp, fmt)
    hsp_file = _format_path(hsp, fmt)
    sns_file = _format_path(sns, fmt)
    raw_file = _format_path(raw, fmt)
    out_file = _format_path(out, fmt, is_new=True)
    
    mrk_file = marker_avg_file(mrk_path)
    hpi_file = mrk_file.path

    # make sure target path exists
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    elif os.path.exists(out_file):
        if not ui.ask("Overwrite?", "Target File Already Exists at: %r. Should "
                      "it be replaced?" % out_file):
            return
        
    cmd = [os.path.join(_mne_dir, 'bin', 'mne_kit2fiff'),
           '--elp', elp_file,
           '--hsp', hsp_file,
           '--sns', sns_file,
           '--hpi', hpi_file,
           '--raw', raw_file,
           '--out', out_file,
           '--stim', ':'.join('%i' % s for s in stim), # '161:162:163:164:165:166:167:168'
           '--sfreq', sfreq,
           '--aligntol', aligntol,
           '--lowpass', lowpass,
           '--highpass', highpass,
           '--stimthresh', stimthresh,
           ]
    
    _run(cmd)



def _run(cmd, v=True):
    """
    cmd: list of strings
        command that is submitted to subprocess.Popen.
    v : bool
        verbose mode
    
    """
    if v:
        print "> COMMAND:"
        for line in cmd:
            print repr(line)
    
    cmd = [str(c) for c in cmd]
    sp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    
    if v:
        print "\n> OUT:"
        print stdout
        print '\n> ERROR:'
        print stderr



    """
    
    """
    


