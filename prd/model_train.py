from __future__ import print_function

import h5py
import numpy as np
np.random.seed(0)
import pandas as pd

# ----------------------------------------------------------------------

fname = '../valid_pulses.txt'
print('Reading:', fname)
df = pd.read_csv(fname, sep=' ', dtype={'pulse': str, 'valid': str})

df.set_index('pulse', inplace=True)

# ----------------------------------------------------------------------

fname = '../dst_bolo.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

dst = dict()
bolo = dict()
bolo_t = dict()

train_pulses = []
valid_pulses = []

for pulse in f:
    dst[pulse] = f[pulse]['dst'][0]
    bolo[pulse] = np.clip(f[pulse]['bolo'][:]/1e6, 0., None)
    bolo_t[pulse] = f[pulse]['bolo_t'][:]
    print('\r%10s %10.4f %10.4f %10.4f %10d' % (pulse,
                                                dst[pulse],
                                                bolo_t[pulse][0],
                                                bolo_t[pulse][-1],
                                                bolo_t[pulse].shape[0]), end='')
    if pd.isnull(df.loc[pulse,'valid']):
        train_pulses.append(pulse)
    else:
        valid_pulses.append(pulse)
print()

f.close()

print('train_pulses:', len(train_pulses))
print('valid_pulses:', len(valid_pulses))

# ----------------------------------------------------------------------

sample_size = 200
batch_size = 2000

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

from keras.models import *
from keras.layers import *
from keras.optimizers import *

model = Sequential()

model.add(Conv1D(32, 5, activation='relu', input_shape=(sample_size, 56)))
model.add(MaxPooling1D())

model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D())

model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D())

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.summary()

opt = Adam(lr=1e-4)

model.compile(optimizer=opt, loss='binary_crossentropy')

# ----------------------------------------------------------------------

from keras.callbacks import *

class MyCallback(Callback):

    def __init__(self):
        self.min_val_loss = None
        self.min_val_epoch = None
        self.f = open('train.log', 'w')
        self.f.close()

    def log_print(self, s):
        print(s)
        self.f = open('train.log', 'a')
        self.f.write(s+'\n')
        self.f.flush()
        self.f.close()

    def on_train_begin(self, logs=None):
        self.log_print('%-10s %5s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss'))

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        t = time.strftime('%H:%M:%S')
        loss = logs['loss']
        val_loss = logs['val_loss']
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.min_val_epoch = epoch
            self.model.save('model.hdf')
            self.log_print('%-10s %5d %10.6f %10.6f *' % (t, epoch, loss, val_loss))
        else:
            self.log_print('%-10s %5d %10.6f %10.6f' % (t, epoch, loss, val_loss))
        if epoch >= 2*self.min_val_epoch:
            print('Stop training.')
            self.model.stop_training = True

# ----------------------------------------------------------------------

try:
    model.fit_generator(generator(train_pulses),
                        steps_per_epoch=100,
                        epochs=10000,
                        verbose=0,
                        callbacks=[MyCallback()],
                        validation_data=generator(valid_pulses),
                        validation_steps=10)
except KeyboardInterrupt:
    print('\nTraining interrupted.')
