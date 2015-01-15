# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
from nose.tools import eq_, ok_, assert_false

from ..._utils.testing import TempDir
from eelbrain.experiment import TreeModel, FileTree


def test_tree():
    "Test simple formatting in the tree"
    class Tree(TreeModel):
        _templates = dict(apath="/{afield}/",
                          afield=('a1', 'a2', 'a3'),
                          field2=('', 'value'))
        def __init__(self, *a, **kw):
            TreeModel.__init__(self, *a, **kw)
            self._register_compound('cmp', ('afield', 'field2'))

    tree = Tree()
    eq_(tree.get('apath'), '/a1/')
    vs = []
    for v in tree.iter('afield'):
        vs.append(v)
        eq_(tree.get('apath') , '/%s/' % v)
        tree.set(afield='a3')
        eq_(tree.get('afield') , 'a3')
        eq_(tree.get('apath'), '/a3/')

    eq_(vs, ['a1', 'a2', 'a3'])
    eq_(tree.get('afield'), 'a1')

    # test compound
    eq_(tree.get('cmp'), 'a1')
    tree.set(field2='value')
    eq_(tree.get('cmp'), 'a1 value')
    tree.set(field2='')
    eq_(tree.get('cmp'), 'a1')

    # test temporary state
    with tree._temporary_state:
        tree.set(afield='a2')
        eq_(tree.get('afield'), 'a2')
    eq_(tree.get('afield'), 'a1')


def test_file_tree():
    "Test file management tree"
    class Tree(FileTree):
        _templates = {'a-folder': '{root}/{folder}',
                      'a-file': '{a-folder}/{name}.txt',
                      'folder': ('f1', 'f2'),
                      'name': ('a1', 'a2', 'a3')}

        def __init__(self, *args, **kwargs):
            FileTree.__init__(self, *args, **kwargs)
            self._bind_make('a-file', self._make_a)

        def load_a(self):
            with open(self.get('a-file', make=True)) as fid:
                return fid.read()

        def _make_a(self):
            with open(self.get('a-file', mkdir=True), 'w') as fid:
                fid.write(self.format("{folder} {name}"))

    root = TempDir()
    tree = Tree(root=root)
    for _ in tree.iter_temp('a-file'):
        eq_(tree.load_a(), tree.format("{folder} {name}"))

    for i, fname in enumerate(tree.iter_temp('a-file')):
        ok_(os.path.exists(fname))
    eq_(i, 5)

    tree.rm('a-file', name='*', folder='f1', confirm=True)
    for fname in tree.iter_temp('a-file', folder='f1'):
        ok_(fname[-6:-4], tree.get('name'))
        assert_false(os.path.exists(fname))
    for fname in tree.iter_temp('a-file', folder='f2'):
        ok_(fname[-6:-4], tree.get('name'))
        ok_(os.path.exists(fname))
