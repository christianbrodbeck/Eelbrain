from eelbrain import *

from eelbrain._wxgui import get_app

if __name__ == '__main__':
    get_app()
    ds = datasets.get_uts(True, nrm=True)
    # res = testnd.ttest_rel('utsnd', 'A', match='rm', ds=ds, samples=100)
    res2 = testnd.anova('utsnd', 'A*B*nrm(A)', ds=ds, samples=100, pmin=0.5)
