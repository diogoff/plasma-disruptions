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
bolo = np.clip(f[pulse]['bolo'][:]/1e6, 0., None)
bolo_t = f[pulse]['bolo_t'][:]

print('dst:', dst)
print('bolo:', bolo.shape, bolo.dtype)
print('bolo_t:', bolo_t.shape, bolo_t.dtype)

f.close()

# ----------------------------------------------------------------------

t0 = 48.6
t1 = 54.0

i0 = np.argmin(np.fabs(bolo_t - t0))
i1 = np.argmin(np.fabs(bolo_t - t1))

bolo = bolo[i0:i1+1]
bolo_t = bolo_t[i0:i1+1]

print('bolo:', bolo.shape, bolo.dtype)
print('bolo_t:', bolo_t.shape, bolo_t.dtype)

# ----------------------------------------------------------------------

from keras.models import *

fname = 'prd/model.hdf'
print('Reading:', fname)
prd_model = load_model(fname)

fname = 'ttd/model.hdf'
print('Reading:', fname)
ttd_model = load_model(fname)

# ----------------------------------------------------------------------

sample_size = 200

X_pred = []
t_pred = []

for i in range(sample_size, bolo.shape[0] + 1):
    x = bolo[i-sample_size:i]
    t = bolo_t[i-1]
    X_pred.append(x)
    t_pred.append(t)

X_pred = np.array(X_pred, dtype=np.float32)
t_pred = np.array(t_pred, dtype=np.float32)

print('X_pred:', X_pred.shape, X_pred.dtype)
print('t_pred:', t_pred.shape, t_pred.dtype)

# ----------------------------------------------------------------------

prd_pred = prd_model.predict(X_pred, batch_size=X_pred.shape[0], verbose=1)
ttd_pred = ttd_model.predict(X_pred, batch_size=X_pred.shape[0], verbose=1)

prd_pred = np.squeeze(prd_pred)
ttd_pred = np.squeeze(ttd_pred)

print('prd_pred:', prd_pred.shape, prd_pred.dtype)
print('ttd_pred:', ttd_pred.shape, ttd_pred.dtype)

# ----------------------------------------------------------------------

fig, ax1 = plt.subplots()

ax1.plot(t_pred, ttd_pred, 'C0', linewidth=1.)
ax1.set_xlabel('(s)')
ax1.set_ylabel('ttd', color='C0')
ax1.tick_params('y', colors='C0')
ax1.set_ylim(0.)

ax2 = ax1.twinx()
ax2.plot(t_pred, prd_pred, 'C1', linewidth=1.)
ax2.set_ylabel('prd', color='C1')
ax2.tick_params('y', colors='C1')
ax2.set_ylim(0., 1.)

if dst > 0.:
    plt.axvline(x=dst, color='k', linestyle='--')

plt.tight_layout()
plt.show()