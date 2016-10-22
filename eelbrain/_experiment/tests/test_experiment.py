# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
from nose.tools import eq_, ok_, assert_false

from ..._utils.testing import TempDir
from eelbrain._experiment import TreeModel, FileTree


class Tree(TreeModel):
    _templates = dict(apath="/{afield}/",
                      afield=('a1', 'a2', 'a3'),
                      field2=('', 'value'))

    def __init__(self, **kwargs):
        TreeModel.__init__(self, **kwargs)
        self._register_compound('cmp', ('afield', 'field2'))


def test_tree():
    "Test simple formatting in the tree"
    tree = Tree()
    eq_(tree.get('apath'), '/a1/')
    vs = []
    for v in tree.iter('afield'):
        vs.append(v)
        eq_(tree.get('apath'), '/%s/' % v)
        tree.set(afield='a3')
        eq_(tree.get('afield'), 'a3')
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


class SlaveTree(TreeModel):
    _templates = {'path': '{a}_{b}_{sb_comp}_{slave}'}

    def __init__(self):
        TreeModel.__init__(self)
        self._register_field('a', ('x', 'y'))
        self._register_field('b', ('u', 'v'))
        self._register_compound('ab', ('a', 'b'))
        self._register_slave_field('s', 'a', lambda f: f['a'].upper())
        self._register_compound('sb', ('s', 'b'))
        self._register_slave_field('comp_slave', 'sb', lambda f: f['sb'].upper())


def test_slave_tree():
    tree = SlaveTree()
    eq_(tree.get('ab'), 'x u')
    eq_(tree.get('sb'), 'X u')
    eq_(tree.get('comp_slave'), 'X U')
    tree.set(a='y')
    eq_(tree.get('ab'), 'y u')
    eq_(tree.get('sb'), 'Y u')
    eq_(tree.get('comp_slave'), 'Y U')

    # .iter()
    eq_(tuple(tree.iter(('a', 'b'))),
        (('x', 'u'), ('x', 'v'), ('y', 'u'), ('y', 'v')))
    eq_(tuple(tree.iter(('b', 'a'))),
        (('u', 'x'), ('u', 'y'), ('v', 'x'), ('v', 'y')))


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
