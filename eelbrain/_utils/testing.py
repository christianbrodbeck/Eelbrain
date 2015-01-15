# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import shutil
import tempfile


class TempDir(str):
    "After MNE-Python mne.utils"
    def __new__(cls):
        return str.__new__(cls, tempfile.mkdtemp())

    def __del__(self):
        shutil.rmtree(self, ignore_errors=True)