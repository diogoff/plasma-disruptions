from __future__ import print_function

import os
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

fname = 'dst_bolo.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

pulses = np.array(sorted(f.keys()))
print('pulses:', len(pulses))

# ----------------------------------------------------------------------

N = 10

r = np.arange(len(pulses))

i_train = r[(r % N) <= N-3]
i_valid = r[(r % N) == N-2]
i_test = r[(r % N) == N-1]

train_pulses = [pulse for pulse in pulses[i_train]]
valid_pulses = [pulse for pulse in pulses[i_valid]]
test_pulses = [pulse for pulse in pulses[i_test]]

print('train_pulses:', len(train_pulses))
print('valid_pulses:', len(valid_pulses))
print('test_pulses:', len(test_pulses))

# ----------------------------------------------------------------------

dirs = ['images',
        os.path.join('images', 'disruptive'),
        os.path.join('images', 'non-disruptive')]
        
for d in dirs:
    if not os.path.isdir(d):
        print('Creating:', d)
        os.mkdir(d)

# ----------------------------------------------------------------------

from keras.models import *

fname = 'prd/model.hdf'
print('Reading:', fname)
prd_model = load_model(fname)

fname = 'ttd/model.hdf'
print('Reading:', fname)
ttd_model = load_model(fname)

# ----------------------------------------------------------------------

fname = 'dst_pred.hdf'
print('Writing:', fname)
fout = h5py.File(fname, 'w')

sample_size = 200

for pulse in test_pulses:

    dst = f[pulse]['dst'][0]
    bolo = f[pulse]['bolo'][:]
    bolo_t = f[pulse]['bolo_t'][:]
    print('%8s %8.4f %8.4f %8.4f %8d' % (pulse,
                                         dst,
                                         bolo_t[0],
                                         bolo_t[-1],
                                         bolo_t.shape[0]))

    X_batch = []
    t_batch = []

    for i in range(sample_size, bolo.shape[0] + 1):
        x = bolo[i-sample_size:i]
        t = bolo_t[i-1]
        X_batch.append(x)
        t_batch.append(t)

    X_batch = np.array(X_batch, dtype=np.float32)
    t_batch = np.array(t_batch, dtype=np.float32)
    
    print('X_batch:', X_batch.shape, X_batch.dtype)
    print('t_batch:', t_batch.shape, t_batch.dtype)
    
    prd_batch = prd_model.predict(X_batch, batch_size=X_batch.shape[0], verbose=0)
    ttd_batch = ttd_model.predict(X_batch, batch_size=X_batch.shape[0], verbose=0)
    
    prd_batch = np.squeeze(prd_batch)
    ttd_batch = np.squeeze(ttd_batch)
    
    print('prd_batch:', prd_batch.shape, prd_batch.dtype)
    print('ttd_batch:', ttd_batch.shape, ttd_batch.dtype)

    g = fout.create_group(pulse)
    g.create_dataset('dst', data=[dst])
    g.create_dataset('prd', data=prd_batch)
    g.create_dataset('ttd', data=ttd_batch)
    g.create_dataset('prd_t', data=t_batch)
    g.create_dataset('ttd_t', data=t_batch)
    
    fig, ax1 = plt.subplots()

    ax1.plot(t_batch, ttd_batch, 'C0', linewidth=1.)
    ax1.set_xlabel('t (s)')
    ax1.set_ylabel('ttd (s)', color='C0')
    ax1.tick_params('y', colors='C0')
    ax1.set_ylim(0.)

    ax2 = ax1.twinx()
    ax2.plot(t_batch, prd_batch, 'C1', linewidth=1.)
    ax2.set_ylabel('prd', color='C1')
    ax2.tick_params('y', colors='C1')
    ax2.set_ylim(0., 1.)

    if dst > 0.:
        plt.axvline(x=dst, color='k', linestyle='--')
        plt.title('pulse %s (disruption @ t=%.4fs)' % (pulse, dst))
        fname = 'images/disruptive/%s.png' % pulse
    else:
        plt.title('pulse %s' % pulse)
        fname = 'images/non-disruptive/%s.png' % pulse

    print('Writing:', fname)
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    plt.close()   

# ----------------------------------------------------------------------

f.close()
fout.close()
