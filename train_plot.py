from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2)

for (k, fname) in enumerate(['ttd/train.log', 'prd/train.log']):

    print('Reading:', fname)
    df = pd.read_csv(fname)

    epoch = df['epoch'].values
    loss = df['loss'].values
    val_loss = df['val_loss'].values

    ax[k].plot(epoch, loss, label='loss')
    ax[k].plot(epoch, val_loss, label='val_loss')
    
    ax[k].set_xlabel('epoch')

    ax[k].legend()
    ax[k].grid()

    i = np.argmin(val_loss)
    min_val_loss = val_loss[i]
    min_val_epoch = epoch[i]

    print('min_val_loss: %10.6f' % min_val_loss)
    print('min_val_epoch: %d' % min_val_epoch)

    (x_min, x_max) = ax[k].get_xlim()
    (y_min, y_max) = ax[k].get_ylim()

    ax[k].plot([x_min, min_val_epoch], [min_val_loss, min_val_loss], 'k--')
    ax[k].plot([min_val_epoch, min_val_epoch], [y_min, min_val_loss], 'k--')

    ax[k].set_xlim(0, x_max)
    ax[k].set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
