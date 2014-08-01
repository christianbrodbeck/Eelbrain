# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from nose.tools import assert_equal

from eelbrain.experiment import TreeModel


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
    assert_equal(tree.get('apath'), '/a1/')
    vs = []
    for v in tree.iter('afield'):
        vs.append(v)
        assert_equal(tree.get('apath') , '/%s/' % v)
        tree.set(afield='a3')
        assert_equal(tree.get('afield') , 'a3')
        assert_equal(tree.get('apath'), '/a3/')

    assert_equal(vs, ['a1', 'a2', 'a3'])
    assert_equal(tree.get('afield'), 'a1')

    # test compound
    assert_equal(tree.get('cmp'), 'a1')
    tree.set(field2='value')
    assert_equal(tree.get('cmp'), 'a1 value')
    tree.set(field2='')
    assert_equal(tree.get('cmp'), 'a1')

    # test temporary state
    with tree._temporary_state:
        tree.set(afield='a2')
        assert_equal(tree.get('afield'), 'a2')
    assert_equal(tree.get('afield'), 'a1')
