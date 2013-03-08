'''
Created on Jun 9, 2012

@author: christian
'''
import fnmatch
import os

import numpy as np

from .. import ui
from ..utils import subp
from ..vessels.data import dataset, var

__all__ = ['Edf']



class Edf(object):
    """
    Class for reading an eyelink .edf file.

    Reads an eyelink .edf file and extracts epoch acceptability
    based on contamination with ocular artifacts (saccades and blinks).
    An edf file reader is initialized with the path to the corresponding
    file::

        >>> path = '/path/to/edf.edf'
        >>> edf = load.eyelink.Edf(path)

    There are several ways of retrieving trial acceptability values (see the
    used methods' documentation for more information):


    1. **For all triggers:**
       Acceptability for all triggers can be added to a dataset with a single
       command if the dataset contains the same events as the edf file::

          >>> edf.mark_all(ds, ...)


    2. **For a subset of triggers:**
       Often it is more efficient to compute acceptability only for a subset of
       the triggers contained in the edf file. For those cases, the trigger time
       should first be added to the complete dataset with::

           >>> edf.add_T_to(ds)

       Now, the dataset can be decimated::

           >>> ds = ds.subset(...)

       and acceptability can be added for the subset::

           >>> edf.mark(ds, ...)

    3. **Customized:**
       For a more customized use, all triggers from the edf can be retrieved using::

           >>> ds_edf = edf.get_triggers()

       The ``ds_edf['t_edf']`` time variable can be used to add trigger time values to
       arbitrary events, which can then in turn be used with::

           >>> edf.mark(...)

    """
    def __init__(self, path=None):
        """
        Parameters
        ----------
        path : str(path) | None
            Path to the .edf file. The If path contains '*', the files matching
            the pattern are concatenated. If None, a file-open dialogue will be
            displayed.

        """
        if path is None:
            path = ui.ask_file("Load an eyelink .edf file", "Pick the edf file",
                               ext=[('edf', 'eyelink data format')])

        # find all paths from which to read
        self.path = path
        if '*' in path:
            head, tail = os.path.split(path)
            if '*' in head:
                err = ("Invalid path: %r. All edf files need to be in the same "
                       "directory." % path)
                raise ValueError(err)

            fnames = sorted(fnmatch.filter(os.listdir(head), tail))
            self.paths = [os.path.join(head, fname) for fname in fnames]
        else:
            self.paths = [path]

        triggers = []
        artifacts = []
        for path in self.paths:
            edf = subp.edf_file(path)
            triggers += edf.triggers
            artifacts += edf.artifacts

        dtype = [('T', np.uint32), ('Id', np.uint8)]
        self.triggers = np.array(triggers, dtype=dtype)
        dtype = np.dtype([('event', np.str_, 6), ('start', np.uint32), ('stop', np.uint32)])
        self.artifacts = np.array(artifacts, dtype=dtype)

    def __repr__(self):
        return "Edf(%r)" % self.path

    def assert_Id_match(self, ds=None, Id='eventId'):
        """
        Make sure the Edf and another event list describe the same events.

        Raises an error if the Ids in ``Id`` do not match the Ids in the Edf
        file(s).

        Parameters
        ----------
        ds : None | dataset
            Dataset with events.
        Id : str | array
            If `ds` is a dataset, `Id` should be a string naming the variable
            in `ds` containing the event IDs. If `ds` is None, `Id` should be
            a series of event Ids.

        """
        if isinstance(Id, str):
            Id = ds[Id]

        ID_edf = self.triggers['Id']
        if len(Id) != len(ID_edf):
            lens = (len(Id), len(ID_edf))
            mm = min(lens)
            for i in xrange(mm):
                if Id[i] != ID_edf[i]:
                    mm = i
                    break

            args = (getattr(ds, 'name', 'None'), self.path) + lens + (mm,)
            err = ("dataset %r containes different number of events from edf "
                   "file %r (%i vs %i); first mismatch at %i." % args)
            raise ValueError(err)

        check = (Id == ID_edf)
        if not all(check):
            err = "Event ID mismatch: %s" % np.where(check == False)[0]
            raise ValueError(err)

    def add_T_to(self, ds, Id='eventID', t_edf='t_edf'):
        """
        Add edf trigger times as a variable to dataset ds.
        These can then be used for Edf.add_by_T(ds) after ds hads been
        decimated.

        Parameters
        ----------
        ds : dataset
            The dataset to which the variable is added
        Id : str | var | None
            variable (or its name in the dataset) containing event IDs. Values
            in this variable are checked against the events in the EDF file,
            and an error is raised if there is a mismatch. This test can be
            skipped by setting Id=None.
        t_edf : str
            Name for the target variable holding the edf trigger times.

        """
        if Id:
            self.assert_Id_match(ds=ds, Id=Id)
            if isinstance(Id, str):
                Id = ds[Id]

        ds[t_edf] = var(self.triggers['T'])

    def filter(self, ds, tstart= -0.1, tstop=0.6, use=['ESACC', 'EBLINK'],
               T='t_edf'):
        """
        Return a copy of the dataset ``ds`` with all bad events removed. A
        dataset containing all the bad events is stored in
        ``ds.info['rejected']``.

        Parameters
        ----------
        ds : dataset
            The dataset that is to be filtered
        tstart : scalar
            Start of the time window in which to look for artifacts
        tstop : scalar
            End of the time window in which to look for artifacts
        use : list of str
            Events which are to be treated as artifacts ('ESACC' and 'EBLINK')
        T : str | var
            Variable describing edf-relative timing for the events in ``ds``.
            Usually this is a string key for a variable in ``ds``.

        """
        if isinstance(T, str):
            T = ds[T]
        accept = self.get_accept(T, tstart=tstart, tstop=tstop, use=use)
        accepted = ds.subset(accept)
        rejected = ds.subset(accept == False)
        accepted.info['rejected'] = rejected
        return accepted

    def get_accept(self, T=None, tstart= -0.1, tstop=0.6, use=['ESACC', 'EBLINK']):
        """
        returns a boolean var indicating for each epoch whether it should be
        accepted or not based on ocular artifacts in the edf file.

        Parameters
        ----------
        T : array-like | None
            List of time points (in the edf file's time coordinates). If None,
            the edf's trigger events are used.
        tstart : scalar
            start of the epoch relative to the event (in seconds)
        tstop : scalar
            end of the epoch relative to the even (in seconds)

        """
        if T is None:
            T = self.triggers['T']

        # conert to ms
        start = int(tstart * 1000)
        stop = int(tstop * 1000)

        self._debug = []

        # get data for triggers
        N = len(T)
        accept = np.empty(N, np.bool_)

        X = tuple(self.artifacts['event'] == name for name in use)
        idx = np.any(X, axis=0)
        artifacts = self.artifacts[idx]

        for i, t in enumerate(T):
            starts_before_tstop = artifacts['start'] < t + stop
            stops_after_tstart = artifacts['stop'] > t + start
            overlap = np.all((starts_before_tstop, stops_after_tstart), axis=0)
            accept[i] = not np.any(overlap)

            self._debug.append(overlap)

        return accept

    def get_T(self, name='t_edf'):
        "returns all trigger times in the dataset"
        return var(self.triggers['T'], name=name)

    def get_triggers(self, Id='Id', T='t_edf'):
        """
        Returns a dataset with trigger Ids and corresponding Edf time values

        """
        ds = dataset()
        ds[Id] = var(self.triggers['Id'])
        ds[T] = self.get_T(name=T)
        return ds

    def mark(self, ds, tstart= -0.1, tstop=0.6, good=None, bad=False,
             use=['ESACC', 'EBLINK'], T='t_edf', target='accept'):
        """
        Mark events in ds as acceptable or not. ds needs to contain edf trigger
        times in a variable whose name is specified by the ``T`` argument.

        Parameters
        ----------
        ds : dataset
            dataset that contains the data to work with.
        tstart : scalar
            start of the time window relevant for rejection.
        tstop : scalar
            stop of the time window relevant for rejection.
        good : bool | None
            vale assigned to epochs that should be retained based on
            the eye-tracker data.
        bad : bool | None
            value that is assigned to epochs that should be rejected
            based on the eye-tracker data.
        use : list of str
            Artifact categories to include
        T : var
            variable providing the trigger time values
        target : var
            variable to which the good/bad values are assigned (if it does not
            exist, a new variable will be created with all values True
            initially)

        """
        if isinstance(target, str):
            if target in ds:
                target = ds[target]
            else:
                ds[target] = target = var(np.ones(ds.n_cases, dtype=bool))

        if isinstance(T, str):
            T = ds[T]

        accept = self.get_accept(T, tstart=tstart, tstop=tstop, use=use)
        if good is not None:
            target[accept] = good
        if bad is not None:
            target[np.invert(accept)] = bad

    def mark_all(self, ds, tstart= -0.1, tstop=0.6, good=None, bad=False,
                 use=['ESACC', 'EBLINK'],
                 Id='eventID', target='accept'):
        """
        Mark each epoch in the ds for acceptability based on overlap with
        blinks and saccades. ds needs to contain the same number of triggers
        as the edf file. For adding acceptability to a decimated ds, use
        Edf.add_T_to(ds, ...) and then Edf.mark(ds, ...).

        Parameters
        ----------
        ds : dataset
            dataset that contains the data to work with.
        tstart : scalar
            start of the time window relevant for rejection.
        tstop : scalar
            stop of the time window relevant for rejection.
        good :
            vale assigned to epochs that should be retained based on
            the eye-tracker data.
        bad :
            value that is assigned to epochs that should be rejected
            based on the eye-tracker data.
        use : list of str
            Artifact categories to include
        Id : var | None
            variable containing trigger Ids for asserting that the dataset
            contains the same triggers as rhe edf file(s). The test can be
            skipped by setting Id=None
        target : var
            variable to which the good/bad values are assigned (if it does not
            exist, a new variable will be created with all values True
            initially)

        """
        if Id:
            self.assert_Id_match(ds=ds, Id=Id)
            if isinstance(Id, str):
                Id = ds[Id]

        T = self.get_T()
        self.mark(ds, tstart=tstart, tstop=tstop, good=good, bad=bad, use=use,
                  T=T, target=target)
