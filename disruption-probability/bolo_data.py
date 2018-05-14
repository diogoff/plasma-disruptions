from __future__ import print_function

import h5py
import numpy as np
from ppf import *

ppfgo()
ppfuid('jetppf', 'r')
ppfssr(i=[0,1,2,3,4])

# -------------------------------------------------------------------------------------

def get_data(pulse, dda, dtyp):
    ihdata, iwdata, data, x, t, ier = ppfget(pulse, dda, dtyp, reshape=1)
    if (ier != 0) or (len(data) < 2) or (len(t) < 2):
        raise ValueError
    return data, t

def get_bolo(pulse):
    kb5h, kb5h_t = get_data(pulse, 'bolo', 'kb5h')
    kb5v, kb5v_t = get_data(pulse, 'bolo', 'kb5v')
    assert np.all(kb5h_t == kb5v_t)
    bolo = np.hstack((kb5h, kb5v))
    bolo_t = kb5h_t
    return bolo, bolo_t

# -------------------------------------------------------------------------------------

pulse0 = 80128
pulse1 = 92504

fname = 'bolo_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in range(pulse0, pulse1+1):
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
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)

    print('%10s %10.4f %10.4f %10.4f %10d %10d' % (pulse,
                                                   bolo_t[0],
                                                   bolo_t[-1],
                                                   step,
                                                   n,
                                                   bolo_t.shape[0]))

f.close()
