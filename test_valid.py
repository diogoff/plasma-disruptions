from __future__ import print_function

import os
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

fname = 'train_data.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

pulses = np.array(f.keys())
print('pulses:', len(pulses))

# ----------------------------------------------------------------------

r = np.arange(len(pulses))

N = 10

i_train = ((r % N) >= 1)
i_valid = ((r % N) == 0)

train_pulses = list(pulses[i_train])
valid_pulses = list(pulses[i_valid])

print('train_pulses:', len(train_pulses))
print('valid_pulses:', len(valid_pulses))

# ----------------------------------------------------------------------

dirs = ['images',
        os.path.join('images', 'disruptive'),
        os.path.join('images', 'non-disruptive')]
        
for d in dirs:
    if not os.path.isdir(d):
        print('Creating:', d)
        os.mkdir(d)

# ----------------------------------------------------------------------

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# ----------------------------------------------------------------------

import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

from keras.backend.tensorflow_backend import set_session

set_session(tf.Session(config=config))

# ----------------------------------------------------------------------

from keras.models import *

fname = 'prd/model.hdf'
print('Reading:', fname)
prd_model = load_model(fname)

fname = 'ttd/model.hdf'
print('Reading:', fname)
ttd_model = load_model(fname)

# ----------------------------------------------------------------------

fname = 'test_valid.hdf'
print('Writing:', fname)
fout = h5py.File(fname, 'w')

sample_size = 200

for pulse in valid_pulses:

    dst = f[pulse]['dst'][0]
    bolo = np.clip(f[pulse]['bolo'][:]/1e6, 0., None)
    bolo_t = f[pulse]['bolo_t'][:]

    print('%8s %8.4f %8.4f %8.4f %8d' % (pulse,
                                         dst,
                                         bolo_t[0],
                                         bolo_t[-1],
                                         bolo_t.shape[0]), end='\t')

    X_batch = []
    t_batch = []

    for i in range(sample_size, bolo.shape[0] + 1):
        x = bolo[i-sample_size:i]
        t = bolo_t[i-1]
        X_batch.append(x)
        t_batch.append(t)

    X_batch = np.array(X_batch, dtype=np.float32)
    t_batch = np.array(t_batch, dtype=np.float32)
    
    prd_batch = prd_model.predict(X_batch, batch_size=X_batch.shape[0], verbose=0)
    ttd_batch = ttd_model.predict(X_batch, batch_size=X_batch.shape[0], verbose=0)
    
    prd_batch = np.squeeze(prd_batch)
    ttd_batch = np.squeeze(ttd_batch)
    
    g = fout.create_group(pulse)
    g.create_dataset('dst', data=[dst])
    g.create_dataset('prd', data=prd_batch)
    g.create_dataset('ttd', data=ttd_batch)
    g.create_dataset('prd_t', data=t_batch)
    g.create_dataset('ttd_t', data=t_batch)
    
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
