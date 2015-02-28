# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from nose.tools import eq_
from numpy.testing import assert_array_equal

from eelbrain import datasets, load, set_log_level
from eelbrain._utils.testing import TempDir
from eelbrain._wxgui.select_epochs import Document, Model


def test_select_epochs():
    "Test Select-Epochs GUI Document"
    set_log_level('warning', 'mne')
    ds = datasets.get_mne_sample(sns=True)
    tempdir = TempDir()
    path = os.path.join(tempdir, 'rej.pickled')

    # create a file
    doc = Document(ds, 'sns')
    doc.set_path(path)
    doc.set_case(1, False, 'tag')
    doc.set_bad_channels([1])
    # check modifications
    eq_(doc.accept[1], False)
    eq_(doc.tag[1], 'tag')
    eq_(doc.bad_channels, [1])
    doc.save()

    # check the file
    ds_ = load.unpickle(path)
    eq_(doc.epochs.sensor.dimindex(ds_.info['bad_channels']), [1])

    # reload the file
    doc = Document(ds, 'sns', path=path)
    eq_(doc.accept[1], False)
    assert_array_equal(doc.accept[2:], True)
    eq_(doc.tag[1], 'tag')
    eq_(doc.bad_channels, [1])

    # create a model
    model = Model(doc)

    # accept
    model.set_case(0, False)
    eq_(doc.accept[0], False)
    model.history.undo()
    eq_(doc.accept[0], True)
    model.history.redo()
    eq_(doc.accept[0], False)

    # bad channels
    model.set_bad_channels([1, 10])
    eq_(doc.bad_channels, [1, 10])
    model.history.undo()
    eq_(doc.bad_channels, [1])
    model.history.redo()
    eq_(doc.bad_channels, [1, 10])

    # reload to reset
    model.load(path)
    eq_(doc.accept[0], True)
    eq_(doc.bad_channels, [1])
