from __future__ import print_function

import h5py
import numpy as np
np.random.seed(0)

# ----------------------------------------------------------------------

fname = '../train_data.hdf'
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

dst = dict()
bolo = dict()
bolo_t = dict()

for pulse in train_pulses + valid_pulses:
    dst[pulse] = f[pulse]['dst'][0]
    bolo[pulse] = np.clip(f[pulse]['bolo'][:]/1e6, 0., None)
    bolo_t[pulse] = f[pulse]['bolo_t'][:]
    print('\r%10s %10.4f %10.4f %10.4f %10d' % (pulse,
                                                dst[pulse],
                                                bolo_t[pulse][0],
                                                bolo_t[pulse][-1],
                                                bolo_t[pulse].shape[0]), end='')

print()

f.close()

# ----------------------------------------------------------------------

sample_size = 200

def generator(batch_size):
    X_batch = []
    Y_batch = []
    while True:
        pulse = np.random.choice(train_pulses)
        if bolo[pulse].shape[0] < sample_size:
            continue
        i = np.random.randint(sample_size, bolo[pulse].shape[0] + 1)
        x = bolo[pulse][i-sample_size:i]
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

X_valid = []
Y_valid = []

for pulse in valid_pulses:
    for i in range(sample_size, bolo[pulse].shape[0] + 1, sample_size):
        x = bolo[pulse][i-sample_size:i]
        y = 1. if dst[pulse] > 0. else 0.
        X_valid.append(x)
        Y_valid.append(y)

X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

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
from keras.layers import *

model = Sequential()

model.add(Conv1D(32, 5, input_shape=(sample_size, 56)))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(64, 5))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(Conv1D(128, 5))
model.add(Activation('relu'))
model.add(MaxPooling1D())

model.add(LSTM(128))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# ----------------------------------------------------------------------

from keras.optimizers import *

opt = Adam(lr=1e-4)

model.compile(optimizer=opt, loss='binary_crossentropy')

# ----------------------------------------------------------------------

from keras.callbacks import *

class MyCallback(Callback):
    
    def on_train_begin(self, logs=None):
        self.min_val_loss = None
        self.min_val_epoch = None
        self.min_val_weights = None
        fname = 'train.log'
        print('Writing:', fname)
        self.log = open(fname, 'w')
        print('%-10s %10s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss'))
        self.log.write('epoch,loss,val_loss\n')
        self.log.flush()
        
    def on_epoch_end(self, epoch, logs=None):
        t = time.strftime('%H:%M:%S')
        loss = logs['loss']
        val_loss = logs['val_loss']
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.min_val_epoch = epoch
            self.min_val_weights = self.model.get_weights()
            print('%-10s %10d %10.6f %10.6f *' % (t, epoch, loss, val_loss))
        else:
            print('%-10s %10d %10.6f %10.6f' % (t, epoch, loss, val_loss))
        self.log.write('%d,%f,%f\n' % (epoch, loss, val_loss))
        self.log.flush()
        if epoch > 2*self.min_val_epoch:
            print('Stop training.')
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.log.close()

    def get_weights(self):
        return self.min_val_weights

# ----------------------------------------------------------------------

batch_size = 2000
steps_per_epoch = 50
epochs = 10000
verbose = 0

print('batch_size:', batch_size)

mc = MyCallback()

try:
    model.fit_generator(generator(batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=[mc],
                        validation_data=(X_valid, Y_valid),
                        workers=8,
                        max_queue_size=100,
                        use_multiprocessing=True)

except KeyboardInterrupt:
    print('\nTraining interrupted.')

print('Loading weights.')
model.set_weights(mc.get_weights())

# ----------------------------------------------------------------------

fname = 'model.hdf'
print('Writing:', fname)
model.save(fname)
