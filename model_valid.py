from __future__ import print_function

import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

from keras.models import *

fname = 'prd/model.hdf'
print('Reading:', fname)
prd = load_model(fname)

fname = 'ttd/model.hdf'
print('Reading:', fname)
ttd = load_model(fname)

# ----------------------------------------------------------------------

fname = 'dst_bolo.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

sample_size = 200

for pulse in f:

    if int(pulse) < 91368:
        continue

    dst = f[pulse]['dst'][0]
    bolo = np.clip(f[pulse]['bolo'][:]/1e6, 0., None)
    bolo_t = f[pulse]['bolo_t'][:]
    print('%10s %10.4f %10.4f %10.4f %10d' % (pulse,
                                              dst,
                                              bolo_t[0],
                                              bolo_t[-1],
                                              bolo_t.shape[0]))
    X_batch = []
    t_batch = []
    for i in range(sample_size, bolo.shape[0]):
        x = bolo[i-sample_size+1:i+1]
        t = bolo_t[i]
        X_batch.append(x)
        t_batch.append(t)
    X_batch = np.array(X_batch, dtype=np.float32)
    t_batch = np.array(t_batch, dtype=np.float32)
    prd_batch = prd.predict(X_batch, batch_size=X_batch.shape[0], verbose=0)
    ttd_batch = ttd.predict(X_batch, batch_size=X_batch.shape[0], verbose=0)

    fig, ax1 = plt.subplots()

    ax1.plot(t_batch, ttd_batch, 'b')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('ttd', color='b')
    ax1.tick_params('y', colors='b')        
    ax1.set_ylim(0.)

    ax2 = ax1.twinx()
    ax2.plot(t_batch, prd_batch, 'r')
    ax2.set_ylabel('prd', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(0., 1.)

    if dst > 0.:
        plt.axvline(x=dst, color='k', linestyle='--')
        plt.title('pulse %s (disruption @ t=%.4fs)' % (pulse, dst))
        fname = '%s_%.4f.png' % (pulse, dst)
    else:
        plt.title('pulse %s' % pulse)
        fname = '%s.png' % pulse

    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()        

f.close()
