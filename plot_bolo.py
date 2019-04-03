from __future__ import print_function

import h5py
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# ----------------------------------------------------------------------

fname = 'train_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

pulse = str(92213)

dst = f[pulse]['dst'][0]
bolo = f[pulse]['bolo'][:]
bolo_t = f[pulse]['bolo_t'][:]

print('dst:', dst)
print('bolo:', bolo.shape, bolo.dtype)
print('bolo_t:', bolo_t.shape, bolo_t.dtype)

f.close()

# ----------------------------------------------------------------------

ii = []

for i in range(bolo_t.shape[0]):
    if bolo_t[i] < 47.75:
        continue
    if bolo_t[i] > 54.25:
        continue
    ii.append(i)

ii = np.array(ii)

print('ii:', ii)

# ----------------------------------------------------------------------

labels = dict()

for j in range(bolo.shape[1]):
    if j < 24:
        labels[j] = 'KB5H%02d' % (j+1)
    else:
        labels[j] = 'KB5V%02d' % (j-24+1)

# ----------------------------------------------------------------------

plt.plot(bolo_t[ii], bolo[ii])

# ----------------------------------------------------------------------

t0 = 48.0
t1 = 53.5

maxima = []
for j in range(bolo.shape[1]):
    i_max = None
    for i in range(bolo_t.shape[0]):
        if bolo_t[i] < t0:
            continue
        if bolo_t[i] > t1:
            continue
        if (i_max == None) or (bolo[i,j] > bolo[i_max,j]):
            i_max = i
    maxima.append(i_max)

maxima = sorted([(maxima[j], j) for j in range(len(maxima)) if bolo[maxima[j],j] > 1.])

x_pos = np.linspace(t0+0.5, t1-0.5, len(maxima))
y_pos = np.linspace(2.5, 17.5, len(maxima))

for k, (i, j) in enumerate(maxima):
    x0 = bolo_t[i]
    y0 = bolo[i,j]
    x1 = x_pos[k]
    y1 = y_pos[k]
    plt.plot(x0, y0, 'k.')
    plt.plot([x0, x1], [y0, y1], 'k', linewidth=0.5)
    plt.text(x1, y1, labels[j], horizontalalignment='center')

# ----------------------------------------------------------------------

plt.xlabel('(s)')
plt.ylabel('(MW m$^{-2}$)')

plt.tight_layout()
plt.show()
