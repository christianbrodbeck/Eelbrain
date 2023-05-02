from eelbrain import *

from eelbrain._wxgui import get_app

if __name__ == '__main__':
    get_app()
    ds = datasets.get_uts(True, nrm=True)
    # res = testnd.TTestRelated('utsnd', 'A', match='rm', ds=ds, samples=100)
    res2 = testnd.ANOVA('utsnd', 'A*B*nrm(A)', data=ds, samples=100, pmin=0.5)
