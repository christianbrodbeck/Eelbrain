# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from eelbrain.experiment import TreeModel


def test_tree():
    "Test simple formatting in the tree"
    class Tree(TreeModel):
        _templates = dict(apath="/{afield}/", afield=('a1', 'a2', 'a3'))

    tree = Tree()
    assert tree.get('apath') == '/a1/'
    vs = []
    for v in tree.iter('afield'):
        vs.append(v)
        assert tree.get('apath') == '/%s/' % v
        tree.set(afield='a3')
        assert tree.get('afield') == 'a3'
        assert tree.get('apath') == '/a3/'
    
    assert vs == ['a1', 'a2', 'a3']
    assert tree.get('afield') == 'a1'
