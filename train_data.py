from __future__ import print_function

import h5py
import sqlite3
import numpy as np
import pandas as pd
from ppf import *

ppfgo()
ppfuid('jetppf', 'r')
ppfssr([0,1,2,3,4])

# ----------------------------------------------------------------------

def get_data(pulse, dda, dtyp):
    ihdata, iwdata, data, x, t, ier = ppfget(pulse, dda, dtyp, reshape=1)
    if (ier != 0) or (len(data) < 2) or (len(t) < 2):
        raise ValueError
    return data, t

def get_bolo(pulse):
    kb5h, kb5h_t = get_data(pulse, 'bolo', 'kb5h')
    kb5v, kb5v_t = get_data(pulse, 'bolo', 'kb5v')
    kb5h[:,19] = 0. # broken channel
    kb5v[:,15] = 0. # broken channel
    kb5v[:,22] = 0. # broken channel
    kb5 = np.hstack((kb5h, kb5v))
    kb5 = np.clip(kb5, 0., None) / 1e6 # clip and scale
    assert np.all(kb5h_t == kb5v_t)
    kb5_t = kb5h_t
    return kb5, kb5_t

# ----------------------------------------------------------------------

pulse0 = 80128
pulse1 = 92504

# ----------------------------------------------------------------------

fname = '/home/DISRUPT/DisruptionDatabase/Database/DDB.db'
print('Reading:', fname)
conn = sqlite3.connect(fname)

sql = 'SELECT id, dTime, deliberate FROM JETDDB WHERE id >= %d AND id <= %d' % (pulse0, pulse1)

print('sql:', sql)

df = pd.read_sql_query(sql, conn, index_col='id')

# ----------------------------------------------------------------------

fname = 'train_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in range(pulse0, pulse1+1):

    dst = 0.
    if pulse in df.index:
        if df.loc[pulse,'deliberate'] == 1:
            continue
        dst = df.loc[pulse,'dTime']

    try:
        bolo, bolo_t = get_bolo(pulse)
    except ValueError:
        continue

    t = 40.
    i = np.argmin(np.fabs(bolo_t - t))
    bolo = bolo[i:]
    bolo_t = bolo_t[i:]

    step = round(np.mean(bolo_t[1:] - bolo_t[:-1]), 4)

    n = int(round(0.005/step))
    bolo = np.cumsum(bolo, axis=0)
    bolo = (bolo[n:] - bolo[:-n]) / n
    bolo = bolo[::n]
    bolo_t = bolo_t[n/2+1::n]
    bolo_t = bolo_t[:bolo.shape[0]]

    g = f.create_group(str(pulse))
    g.create_dataset('dst', data=[dst])
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)

    print('%10s %10.4f %10.4f %10.4f %10.4f %10d %10d' % (pulse,
                                                          dst,
                                                          bolo_t[0],
                                                          bolo_t[-1],
                                                          step,
                                                          n,
                                                          bolo_t.shape[0]))

f.close()
