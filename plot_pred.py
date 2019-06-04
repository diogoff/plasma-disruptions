from __future__ import print_function

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# ----------------------------------------------------------------------

if len(sys.argv) < 4:
    print('Usage: %s pulse t0 t1' % sys.argv[0])
    print('Example: %s 92213 48.0 54.0' % sys.argv[0])
    exit()

# ----------------------------------------------------------------------

pulse = sys.argv[1]
print('pulse:', pulse)

t0 = float(sys.argv[2])
print('t0:', t0)

t1 = float(sys.argv[3])
print('t1:', t1)

# ----------------------------------------------------------------------

fname = 'dst_bolo.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

dst = f[pulse]['dst'][0]
bolo = f[pulse]['bolo'][:]
bolo_t = f[pulse]['bolo_t'][:]

print('dst:', dst)
print('bolo:', bolo.shape, bolo.dtype)
print('bolo_t:', bolo_t.shape, bolo_t.dtype)

f.close()

# ----------------------------------------------------------------------

sample_size = 200

i0 = np.argmin(np.fabs(bolo_t - t0))
i1 = np.argmin(np.fabs(bolo_t - t1))

i0 -= sample_size-1
if i0 < 0:
    i0 = 0

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

for i in range(X_pred.shape[0]):
    print('%10.4f %10.3f %10.3f' % (t_pred[i], prd_pred[i], ttd_pred[i]))

# ----------------------------------------------------------------------

fig, ax1 = plt.subplots()

ax1.plot(t_pred, ttd_pred, 'C0', linewidth=1.)
#ax1.axhline(y=1.5, color='C0', linestyle=':')
#xlim = ax1.get_xlim()
#ax1.fill_between([0., 100.], 0, 1.5, color='C0', alpha=0.15)
#ax1.set_xlim(xlim)
ax1.set_xlabel('t (s)')
ax1.set_ylabel('ttd (s)', color='C0')
if dst > 0.:
    ax1.set_title('pulse %s (disruption @ t=%.2fs)' % (pulse, dst), fontsize='medium')
else:
    ax1.set_title('pulse %s (non-disruptive)' % pulse, fontsize='medium')
ax1.tick_params('y', colors='C0')
ax1.set_ylim(0.)

ax2 = ax1.twinx()
ax2.plot(t_pred, prd_pred, 'C1', linewidth=1.)
#ax2.axhline(y=0.85, color='C1', linestyle=':')
#xlim = ax2.get_xlim()
#ax2.fill_between([0., 100.], 0.85, 1., color='C1', alpha=0.15)
#ax2.set_xlim(xlim)
ax2.set_ylabel('prd', color='C1')
ax2.tick_params('y', colors='C1')
ax2.set_ylim(0., 1.)

if dst > 0.:
    plt.axvline(x=dst, color='k', linestyle='--')

plt.tight_layout()
plt.show()
