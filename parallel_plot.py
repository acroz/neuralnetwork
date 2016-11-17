
import numpy as np
from matplotlib import pyplot

cores = []
time = []

with open('timings.txt') as fp:
    for line in fp:
        parts = line.split()
        cores.append(int(parts[0]))
        
        tparts = parts[-2].split(':')
        if len(tparts) == 1:
            time.append(float(tparts[0]))
        else:
            time.append(int(tparts[0]) * 60 + float(tparts[1]))

time = (np.array(time) - 1.153) * (50. / 20.) / 60.

fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)

s = 2./2.5

ax.plot(cores, time, 'x-', color='#4c9ed9', linewidth=2.*s, markersize=8,
        markeredgecolor='#4c9ed9', markeredgewidth=2.*s)

ax.set_ylim(0, None)

ax.set_xlabel('Number of Threads')
ax.set_ylabel('Execution Time (Minutes)')

pyplot.savefig('parallel-scaling.svg')

pyplot.show()
