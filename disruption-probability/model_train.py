from __future__ import print_function

import h5py
import numpy as np
np.random.seed(0)

# ----------------------------------------------------------------------

fname = '../dst_bolo.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

dst = dict()
bolo = dict()
bolo_t = dict()

train_pulses = []
valid_pulses = []

k = 0
for (i, pulse) in enumerate(f):
    dst[pulse] = f[pulse]['dst'][0]
    bolo[pulse] = np.clip(f[pulse]['bolo'][:]/1e6, 0., None)
    bolo_t[pulse] = f[pulse]['bolo_t'][:]
    k += 1
    if k % 10 != 0:
        train_pulses.append(pulse)
        print('%10s %10.4f %10.4f %10.4f %10d' % (pulse,
                                                  dst[pulse],
                                                  bolo_t[pulse][0],
                                                  bolo_t[pulse][-1],
                                                  bolo_t[pulse].shape[0]))
    else:
        valid_pulses.append(pulse)
        print('%10s %10.4f %10.4f %10.4f %10d *' % (pulse,
                                                    dst[pulse],
                                                    bolo_t[pulse][0],
                                                    bolo_t[pulse][-1],
                                                    bolo_t[pulse].shape[0]))

f.close()

n1 = len(train_pulses)
n2 = len([pulse for pulse in train_pulses if dst[pulse] > 0.])

n3 = len(valid_pulses)
n4 = len([pulse for pulse in valid_pulses if dst[pulse] > 0.])

print('train_pulses: %4d' % n1, 'disruptions: %4d (%.2f%%)' % (n2, float(n2)/float(n1)*100.))
print('valid_pulses: %4d' % n3, 'disruptions: %4d (%.2f%%)' % (n4, float(n4)/float(n3)*100.))

# ----------------------------------------------------------------------

sample_size = 200
batch_size = 1024

def generator(pulses):
    X_batch = []
    Y_batch = []
    while True:
        pulse = np.random.choice(pulses)
        if bolo[pulse].shape[0] < sample_size:
            continue
        i = np.random.choice(np.arange(bolo[pulse].shape[0]))
        if i < sample_size:
            continue
        x = bolo[pulse][i-sample_size+1:i+1]
        y = 1. if dst[pulse] > 0. else 0.
        X_batch.append(x)
        Y_batch.append(y)
        if len(X_batch) >= batch_size:
            X_batch = np.array(X_batch, dtype=np.float32)
            Y_batch = np.array(Y_batch, dtype=np.float32)
            yield (X_batch, Y_batch)
            X_batch = []
            Y_batch = []

# ----------------------------------------------------------------------

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)

# ----------------------------------------------------------------------

from model import *
from keras.utils import *
from keras.optimizers import *

with tf.device('/cpu:0'):
    model = create_model(sample_size)

parallel_model = multi_gpu_model(model, gpus=8)

opt = Adam(lr=1e-4)

parallel_model.compile(optimizer=opt, loss='binary_crossentropy')

# ----------------------------------------------------------------------

from keras.callbacks import *

f = open('train.log', 'w')
f.close()

def log_print(s):
    print(s)
    f = open('train.log', 'a')
    f.write(s+'\n')
    f.flush()
    f.close()

class MyCallback(Callback):

    def __init__(self):
        self.min_val_loss = None
        self.min_val_epoch = None

    def on_train_begin(self, logs=None):
        log_print('%-10s %5s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss'))

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        t = time.strftime('%H:%M:%S')
        loss = logs['loss']
        val_loss = logs['val_loss']
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.min_val_epoch = epoch
            model.save_weights('model_weights.hdf', overwrite=True)
            log_print('%-10s %5d %10.6f %10.6f *' % (t, epoch, loss, val_loss))
        else:
            log_print('%-10s %5d %10.6f %10.6f' % (t, epoch, loss, val_loss))
        if epoch >= 2*self.min_val_epoch:
            print('Stop training.')
            parallel_model.stop_training = True

# ----------------------------------------------------------------------

try:
    parallel_model.fit_generator(generator(train_pulses),
                                 steps_per_epoch=100,
                                 epochs=10000,
                                 verbose=0,
                                 callbacks=[MyCallback()],
                                 validation_data=generator(valid_pulses),
                                 validation_steps=10,
                                 workers=8,
                                 use_multiprocessing=True,
                                 max_queue_size=100)
except KeyboardInterrupt:
    print('\nTraining interrupted.')
