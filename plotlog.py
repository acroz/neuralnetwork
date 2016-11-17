
import sys
import numpy as np
from matplotlib import pyplot

def label(column, suffix=''):
    if column == 0:
        return 'Training Set' + suffix
    elif column == 1:
        return 'Cross Validation Set' + suffix

for filename in sys.argv[1:]:

    with open(filename) as fp:
        data = np.loadtxt(fp)
    
    ncol = data.shape[1] - 1

    if len(sys.argv) > 2:
        suff = ' ' + filename
    else:
        suff = ''
    
    for i in range(ncol):
        kwargs = {'label': label(i, suff),
                  'linewidth': 4./2.5,
                  'markersize': 8,
                  'markeredgewidth': 4./2.5}

        if i == 0:
            kwargs['color'] = '#4c9ed9'
            kwargs['markeredgecolor'] = '#4c9ed9'
        elif i == 1:
            kwargs['color'] = '#ac4142'
            kwargs['markeredgecolor'] = '#ac4142'

        pyplot.plot(data[:,0], data[:,i+1], 'x-', **kwargs)

pyplot.xlabel('Hidden Layer Size')
pyplot.ylabel('Accuracy')

pyplot.legend(loc='lower right')
pyplot.show()
