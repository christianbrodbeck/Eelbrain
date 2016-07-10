"""

for permission errors: try ``os.chmod`` or ``os.chown``


subprocess documentation
------------------------

http://docs.python.org/library/subprocess.html
http://www.doughellmann.com/PyMOTW/subprocess/
http://explanatorygap.net/2010/05/10/python-subprocess-over-shell/
http://codeghar.wordpress.com/2011/12/09/introduction-to-python-subprocess-module/


Created on Mar 4, 2012

@author: christian
"""
from __future__ import print_function

import fnmatch
import logging
import os
import re
import shutil
import subprocess
import tempfile

import mne
from mne.utils import get_subjects_dir, run_subprocess

from . import ui


# environment variables that can hold the path to this tool
_env_vars = {'mne': 'MNE_ROOT',
             'freesurfer': 'FREESURFER_HOME',
             'edfapi': 'EYELINK_HOME'}

# bin directory relative to tool location
_bin_dirs = {'mne': 'bin',
             'freesurfer': 'bin',
             'edfapi': ''}

# binaries to look for to test whether the tool directory is set correctly
_test_bins = {'mne': ('mne_add_patch_info', 'mne_analyze'),
              'freesurfer': ('mri_annotation2label',),
              'edfapi': ('edf2asc',)}


def get_root(package):
    "Get the root path for a package"
    if package not in _env_vars:
        raise KeyError("Unknown binary package: %r" % package)
    root = mne.get_config(_env_vars[package])
    if root:
        test_root(package, root)
    else:
        root = _ask_user_for_bin_dir(package)
    return root


def get_bin(package, name):
    "Get path for a binary"
    root = get_root(package)
    bin_path = os.path.join(root, _bin_dirs[package], name)
    if os.path.exists(bin_path):
        return bin_path
    else:
        raise IOError("Binary does not exist: " + bin_path)


def test_root(package, root):
    bin_dir = os.path.join(root, _bin_dirs[package])
    return all(os.path.exists(os.path.join(bin_dir, f)) for f in _test_bins[package])


def _ask_user_for_bin_dir(package):
    """
    Change the location from which binaries are used.

    Parameters
    ----------
    package : str
        Binary package for which to set the directory. One from:
        ``['mne', 'freesurfer', 'edfapi']``
    """
    title = "Select %s Directory" % package
    message = "Please select the directory of the %s package." % package
    while True:
        answer = ui.ask_dir(title, message, must_exist=True)
        if answer:
            if test_root(package, answer):
                mne.set_config(_env_vars[package], answer)
                return answer
            else:
                ui.message("Wrong Directory", "This is not the right directory.")
        else:
            raise IOError("%s directory not set" % package)


_verbose = 1


def command_exists(cmd):
    """
    Return True if the corresponding command exists (based on stackoverflow_
    answer).

    .. _stackoverflow: http://stackoverflow.com/a/11069822/166700

    """
    return subprocess.call(["type", cmd], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) == 0


# classes for files ---

class edf_file:
    """
    Converts an "eyelink data format" (.edf) file to a temporary directory
    and parses its content.

    """
    def __init__(self, path):
        # convert
        if not os.path.exists(path):
            err = "File does not exist: %r" % path
            raise ValueError(err)

        self.source_path = path
        self.temp_dir = tempfile.mkdtemp()
        cmd = [get_bin('edfapi', 'edf2asc'),  # options in Manual p. 106
               '-t',  # use only tabs as delimiters
               '-e',  # outputs event data only
               '-nse',  # blocks output of start events
               '-p', self.temp_dir,  # writes output with same name to <path> directory
               path]

        _run(cmd)

        # find asc file
        name, _ = os.path.splitext(os.path.basename(path))
        ascname = os.path.extsep.join((name, 'asc'))
        self.asc_path = os.path.join(self.temp_dir, ascname)
        self.asc_file = open(self.asc_path)
        self.asc_str = self.asc_file.read()

        # find trigger events
        #                           MSG   time    msg...      ID
        re_trigger = re.compile(r'\bMSG\t(\d+)\tMEG Trigger: (\d+)')
        self.triggers = re_trigger.findall(self.asc_str)

        # find artifacts
        #                            type                    start   end
        re_artifact = re.compile(r'\b(ESACC|EBLINK)\t[LR]\t(\d+)\t(\d+)')
        self.artifacts = re_artifact.findall(self.asc_str)

    def __del__(self):
        shutil.rmtree(self.temp_dir)

    def __repr__(self):
        return 'edf_file(%r)' % self.source_path


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


def open_in_finder(path):
    """
    Open ``path`` in the finder.
    """
    os.system('open %s' % path)


def run_freesurfer_command(command, subjects_dir):
    "Run a FreeSurfer command"
    env = os.environ.copy()
    env['SUBJECTS_DIR'] = subjects_dir

    # find FREESURFER_HOME
    fs_home = mne.get_config('FREESURFER_HOME')
    save_fs_home = False
    while True:
        problem = fs_home_problem(fs_home)
        if problem:
            save_fs_home = True
            print(problem)
            if fs_home == os.environ.get('FREESURFER_HOME', 0):
                print("WARNING: This directory is set as FREESURFER_HOME "
                      "environment variable. As long as you don't remove this "
                      "environment variable, you will be asked for the proper "
                      "FreeSurfer location every time a FreeSurfer command is "
                      "run.")
            message = "Please select the directory where FreeSurfer is installed"
            print(message)
            fs_home = ui.ask_dir("Select FreeSurfer Directory", message)
            if fs_home is False:
                raise RuntimeError("Could not find FreeSurfer")
        else:
            break
    if save_fs_home:
        mne.set_config('FREESURFER_HOME', fs_home)

    # adjust environment
    env['FREESURFER_HOME'] = fs_home
    bin_path = os.path.join(fs_home, 'bin')
    if bin_path not in env['PATH']:
        env['PATH'] = ':'.join((bin_path, env['PATH']))

    # run command
    run_subprocess(command, env=env)


def fs_home_problem(fs_home):
    """Check FREESURFER_HOME path

    Return str describing problem or None if the path is okay.
    """
    if fs_home is None:
        return "The FreeSurfer path is not set"
    elif not os.path.exists(fs_home):
        return "The FreeSurfer path does not exist: %s" % fs_home
    else:
        test_path = os.path.join(fs_home, 'bin', 'mri_surf2surf')
        if not os.path.exists(test_path):
            return ("The FreeSurfer path is invalid (%s does not contain bin/"
                    "mri_surf2surf)" % fs_home)


def process_raw(raw, save='{raw}_filt', args=['projoff'], rm_eve=True, **kwargs):
    """
    Calls ``mne_process_raw`` to process raw fiff files. All
    options can be submitted in ``args`` or as kwargs (for a description of
    the options, see mne manual 2.7.3, ch. 4 / p. 41)

    raw : str(path)
        Source file.
    save : str(path)
        Destination file. ``'{raw}'`` will be replaced with the raw file path
        without the extension.
    args : list of str
        Options that do not require a value (e.g., to specify ``--filteroff``,
        use ``args=['filteroff']``).
    rm_eve : bool
        Remove the event file automatically generated by ``mne_process_raw``
    **kwargs:**
        ``mne_process_raw`` options that require values (see example).

    Example::

        >>> process_raw('/some/file_raw.fif', highpass=8, lowpass=12)

    """
    raw_name, _ = os.path.splitext(raw)
    eve_file = '%s-eve.fif' % raw_name
    if raw_name.endswith('_raw'):
        raw_name = raw_name[:-4]

    save = save.format(raw=raw_name)

    cmd = [get_bin('mne', 'mne_process_raw'),
           '--raw', raw,
           '--save', save]

    for arg in args:
        cmd.append('--%s' % arg)

    for key, value in kwargs.iteritems():
        cmd.append('--%s' % key)
        cmd.append(value)

    _run(cmd)
    if rm_eve:
        try:
            os.remove(eve_file)
        except Exception as err:
            print("Could not remove %r" % eve_file)
            print(err)


def _run(cmd, v=None, cwd=None, block=True):
    """
    cmd: list of strings
        command that is submitted to subprocess.Popen.
    v : 0 | 1 | 2 | None
        verbosity level (0: nothing;  1: stderr;  2: stdout;  None: use
        _verbose module attribute)

    """
    if v is None:
        v = _verbose

    if v > 1:
        print("> COMMAND:")
        for line in cmd:
            print(repr(line))

    cmd = map(unicode, cmd)
    sp = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    if block:
        stdout, stderr = sp.communicate()

        if v > 1:
            print("\n> stdout:")
            print(stdout)

        if v > 0 and stderr:
            print('\n> stderr:')
            print(stderr)

        return stdout, stderr


def setup_mri(subject, subjects_dir=None, ico=4, block=False, redo=False):
    """
    Prepares an MRI for use in the mne-pipeline:

    - runs mne_setup_mri
    - runs mne_setup_source_space
    - runs mne_watershed_bem
    - creates symlinks for watershed files in the bem directory
    - runs mne_setup_forward_model (see MNE manual section 3.7, p. 25)


    Parameters
    ----------

    subject : str
        Subject whose MRI should be processed.
    subjects_dir : None | str(path)
        Overrides `sys.environ['SUBJECTS_DIR']` if not `None`.
    ico : int
        `--ico` argument for `mne_setup_forward_model`.
    block : bool
        Block the Python interpreter until mne_setup_forward_model is
        finished.
    redo : bool
        Run the commands even if the target files already exist.


    .. note::
        The utility needs permission to access the MRI files. In case of a
        permission error, the following shell command can be used on the mri
        folder to set permissions appropriately::

            $ sudo chmod -R 7700 mri-folder

    """
    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
        _sub_dir = None
    else:
        _sub_dir = os.environ.get('SUBJECTS_DIR', None)
        os.environ['SUBJECTS_DIR'] = subjects_dir

    bemdir = os.path.join(subjects_dir, subject, 'bem')

    # mne_setup_mri
    tgt = os.path.join(subjects_dir, subject, 'mri', 'T1', 'neuromag', 'sets',
                       'COR.fif')
    if redo or not os.path.exists(tgt):
        cmd = [get_bin('mne', 'mne_setup_mri'),
               '--subject', subject,
               '--mri', 'T1']
        _run(cmd)

    # mne_setup_source_space
    tgt = os.path.join(bemdir, '%s-ico-%i-src.fif' % (subject, ico))
    if redo or not os.path.exists(tgt):
        cmd = [get_bin('mne', 'mne_setup_source_space'),
               '--subject', subject,
               '--ico', ico]
        _run(cmd)

    # mne_watershed_bem
    tgt = os.path.join(bemdir, 'watershed', '%s_outer_skin_surface' % (subject))
    cmd = [get_bin('mne', 'mne_watershed_bem'),
           '--subject', subject]
    if redo:
        cmd.append('--overwrite')
    _run(cmd)

    # symlinks (MNE-manual 3.6, p. 24)
    for name in ['inner_skull', 'outer_skull', 'outer_skin']:
        src = os.path.join('watershed', '%s_%s_surface' % (subject, name))
        dest = os.path.join(bemdir, '%s.surf' % name)
        if os.path.exists(dest):
            if os.path.islink(dest):
                abs_src = os.path.join(bemdir, src)
                if os.path.realpath(dest) == abs_src:
                    pass
                else:
                    logging.debug("replacing symlink: %r" % dest)
                    os.remove(dest)
                    os.symlink(src, dest)
                    # can raise an OSError: [Errno 13] Permission denied
            else:
                raise IOError("%r exists and is no symlink" % dest)
        else:
            logging.debug("creating symlink: %r" % dest)
            os.symlink(src, dest)

    # mne_setup_forward_model
    cmd = [get_bin('mne', "mne_setup_forward_model"),
           '--subject', subject,
           '--surf',
           '--ico', ico,
           '--homog']

    _run(cmd, block=block)  # -> creates a number of files in <mri_sdir>/bem
    if _sub_dir:
        os.environ['SUBJECTS_DIR'] = _sub_dir


def _run_mne_gui(name, cwd, modal, subject, subjects_dir):
    root = get_root('mne')
    env = os.environ.copy()
    env['MNE_ROOT'] = root
    if subjects_dir is not None:
        env['SUBJECTS_DIR'] = subjects_dir
    if subject:
        env['SUBJECT'] = subject

    setup_path = os.path.join(root, 'bin', 'mne_setup_sh').replace(' ', '\ ')
    setup = '. %s' % setup_path

    cmd = os.path.join(root, 'bin', name).replace(' ', '\ ')
    p = subprocess.Popen(setup + ';' + cmd, shell=True, cwd=cwd, env=env)

    if modal:
        print("Waiting for %s to be closed..." % name)
        p.wait()


def run_mne_analyze(fif_dir, subject=None, subjects_dir=None, modal=False):
    """invokes mne_analyze (e.g., for manual coregistration)

    Parameters
    ----------
    fif_dir : str
        the directory containing the fiff files.
    subject : None | str
        The name of the MRI subject.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    modal : bool
        Causes the shell to block until mne_analyze is closed.


    Notes
    -----
    **Coregistration Procedure** (For more information see MNE-manual 3.10 &
    12.11):

    #. File > Load Surface: select the subject`s directory and "Inflated"
    #. File > Load digitizer data: select the fiff file
    #. View > Show viewer

       a. ``Options``: Switch cortical surface display off, make
          scalp transparent. ``Done``

    #. Adjust > Coordinate Alignment:

       a. set LAP, Nasion and RAP.
       b. ``Align using fiducials``.
       c. (``Omit``)
       d. Run ``ICP alignment`` with 20 steps
       e. ``Save default``

    this creates a file next to the raw file with the '-trans.fif' extension.

    """
    _run_mne_gui('mne_analyze', fif_dir, modal, subject, subjects_dir)


def run_mne_browse_raw(fif_dir, subject=None, subjects_dir=None, modal=False):
    """Invokes mne_browse_raw

    Parameters
    ----------
    fif_dir : str
        the directory containing the fiff files.
    subject : None | str
        The name of the MRI subject.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    modal : bool
        Causes the shell to block until mne_browse_raw is closed.
    """
    _run_mne_gui('mne_browse_raw', fif_dir, modal, subject, subjects_dir)


def do_forward_solution(paths=dict(rawfif=None,
                                   mri_sdir=None,
                                   fwd='{fif}-fwd.fif',
                                   bem=None,
                                   src=None,
                                   trans=None),
                        overwrite=False, v=1):
    """
    MNE Handbook 3.13

    """
    fif_file = paths.get('rawfif')
    mri_sdir = paths.get('mri_sdir')
    fwd_file = paths.get('fwd')
    bem_file = paths.get('bem')
    src_file = paths.get('src')
    trans_file = paths.get('trans')

    mri_dir, mri_subject = os.path.split(mri_sdir)

    fif_name, _ = os.path.splitext(fif_file)
    fwd_file = fwd_file.format(fif=fif_name)

    os.environ['SUBJECTS_DIR'] = mri_dir
    cmd = [get_bin('mne', "mne_do_forward_solution"),
           '--subject', mri_subject,
           '--src', src_file,
           '--bem', bem_file,
#           '--mri', mri_cor_file, # MRI description file containing the MEG/MRI coordinate transformation.
           '--mri', trans_file,  # MRI description file containing the MEG/MRI coordinate transformation.
#           '--trans', trans_file, #  head->MRI coordinate transformation (obviates --mri).
           '--meas', fif_file,  # provides sensor locations and coordinate transformation between the MEG device coordinates and MEG head-based coordinates.
           '--fwd', fwd_file,  # '--destdir', target_file_dir, # optional
           '--megonly']

    if overwrite:
        cmd.append('--overwrite')
    elif os.path.exists(fwd_file):
        raise IOError("fwd file at %r already exists" % fwd_file)

    out, err = _run(cmd, v=v)
    if os.path.exists(fwd_file):
        return fwd_file
    else:
        err = "fwd-file not created"
        err = os.linesep.join([err, "command out:", out])
        raise RuntimeError(err)


def do_inverse_operator(fwd_file, cov_file, inv_file='{cov}inv.fif',
                        loose=False, fixed=False):
    cov, _ = os.path.splitext(cov_file)
    if cov.endswith('cov'):
        cov = cov[:-3]

    inv_file = inv_file.format(cov=cov)

    cmd = [get_bin('mne', "mne_do_inverse_operator"),
           '--fwd', fwd_file,
           '--meg',  # Employ MEG data in the inverse calculation. If neither --meg nor --eeg is set only MEG channels are included.
           '--depth',
           '--megreg', 0.1,
           '--noisecov', cov_file,
           '--inv', inv_file,  # Save the inverse operator decomposition here.
           ]

    if loose:
        cmd.extend(['--loose', loose])
    elif fixed:
        cmd.append('--fixed')

    _, err = _run(cmd)
    if not os.path.exists(inv_file):
        raise RuntimeError(os.linesep.join(["inv-file not created", err]))


def mne_make_morph_maps(s_to='fsaverage', s_from=None, redo=False, subjects_dir=None):
    """
    s_to, s_from : str | None
        None -> all subejcts

    """
    cmd = [get_bin('mne', "mne_make_morph_maps")]
    if not s_to or not s_from:
        cmd.append('--all')
    if s_to:
        cmd.extend(('--to', s_to))
    if s_from:
        cmd.extend(('--from', s_from))
    if redo:
        cmd.appen('--redo')
    if subjects_dir:
        os.environ['SUBJECTS_DIR'] = subjects_dir
    _run(cmd)


# freesurfer---


def _fs_hemis(arg):
    if arg == '*':
        return ['lh', 'rh']
    elif arg in ['lh', 'rh']:
        return [arg]
    else:
        raise ValueError("hemi has to be 'lh', 'rh', or '*' (no %r)" % arg)


def _fs_subjects(arg, exclude=[], subjects_dir=None):
    if '*' in arg:
        subjects_dir = get_subjects_dir(subjects_dir)
        subjects = fnmatch.filter(os.listdir(subjects_dir), arg)
        subjects = filter(os.path.isdir, subjects)
        for subject in exclude:
            if subject in subjects:
                subjects.remove(subject)
    else:
        subjects = [arg]
    return subjects
