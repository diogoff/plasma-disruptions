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

time = []

for i in range(bolo_t.shape[0]):
    if bolo_t[i] < 47.75:
        continue
    if bolo_t[i] > 54.25:
        continue
    time.append(i)

time = np.array(time)

print('time:', time)

# ----------------------------------------------------------------------

plt.plot(bolo_t[time], bolo[time], linewidth=1.)

plt.xlabel('(s)')
plt.ylabel('(MW m$^{-2}$)')

plt.tight_layout()
plt.show()
