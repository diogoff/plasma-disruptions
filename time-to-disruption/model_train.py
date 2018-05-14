from __future__ import print_function

import h5py
import numpy as np
np.random.seed(0)
from tqdm import tqdm

# ----------------------------------------------------------------------

fname = 'dst_bolo.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

dst = dict()
bolo = dict()
bolo_t = dict()

train_pulses = []
valid_pulses = []

for (i, pulse) in tqdm(enumerate(f)):
    if (i+1) % 10 != 0:
        train_pulses.append(pulse)
    else:
        valid_pulses.append(pulse)
    dst[pulse] = f[pulse]['dst'][0]
    bolo[pulse] = f[pulse]['bolo'][:]
    bolo_t[pulse] = f[pulse]['bolo_t'][:]

f.close()

print('train_pulses:', len(train_pulses))
print('valid_pulses:', len(valid_pulses))

# ----------------------------------------------------------------------

def generator(pulses, sample_size, batch_size):
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
        y = dst[pulse] - bolo_t[pulse][i]
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

sample_size = 200

with tf.device('/cpu:0'):
    model = create_model(sample_size)

gpus = 8

parallel_model = multi_gpu_model(model, gpus=gpus)

opt = Adam(lr=1e-4)

parallel_model.compile(optimizer=opt, loss='mae')

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

batch_size = 1000 * gpus

train_generator = generator(train_pulses, sample_size, batch_size)
valid_generator = generator(valid_pulses, sample_size, batch_size)

try:
    parallel_model.fit_generator(train_generator,
                                 steps_per_epoch=100,
                                 epochs=100000,
                                 verbose=1,
                                 callbacks=[MyCallback()],
                                 validation_data=valid_generator,
                                 validation_steps=10)
except KeyboardInterrupt:
    print('Training interrupted.')
