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

def get_bolo(pulse):
    ihdata, iwdata, kb5h, x, kb5h_t, ier = ppfget(pulse, 'bolo', 'kb5h', reshape=1)
    ihdata, iwdata, kb5v, x, kb5v_t, ier = ppfget(pulse, 'bolo', 'kb5v', reshape=1)
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

fname = '/home/DISRUPT/DisruptionDatabase/Database/DDB.db'
print('Reading:', fname)
conn = sqlite3.connect(fname)

sql = 'SELECT id, dTime, deliberate FROM JETDDB WHERE id>=%d AND id<=%d' % (pulse0, pulse1)
print('sql:', sql)

df = pd.read_sql_query(sql, conn, index_col='id')
print('df:', df.shape)

# ----------------------------------------------------------------------

fname = 'dst_bolo.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in range(pulse0, pulse1+1):
    
    dst = 0.
    if (pulse in df.index):
        if df.loc[pulse,'deliberate'] == 1:
            continue
        dst = df.loc[pulse,'dTime']
            
    try:
        bolo, bolo_t = get_bolo(pulse)
    except:
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
