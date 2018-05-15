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

def get_ipla(pulse):
    return get_data(pulse, 'magn', 'ipla')

def get_bolo(pulse):
    kb5h, kb5h_t = get_data(pulse, 'bolo', 'kb5h')
    kb5v, kb5v_t = get_data(pulse, 'bolo', 'kb5v')
    assert np.all(kb5h_t == kb5v_t)
    bolo = np.hstack((kb5h, kb5v))
    bolo_t = kb5h_t
    return bolo, bolo_t

# -------------------------------------------------------------------------------------

def get_dst(ipla, ipla_t):
    x0 = ipla_t[:-1]
    x1 = ipla_t[1:]
    y0 = ipla[:-1]
    y1 = ipla[1:]
    grad = (y1-y0)/(x1-x0)
    grad_t = (x0+x1)/2.
    dst = 0.
    lim = 20e6 # 20 MA/s
    pos = np.where(grad > lim)[0]
    if len(pos) > 0:
        i = pos[0]
        x0 = grad_t[i-1]
        x1 = grad_t[i]
        y0 = grad[i-1]
        y1 = grad[i]
        m = (y1-y0)/(x1-x0)
        b = (y0*x1-y1*x0)/(x1-x0)
        dst = (lim-b)/m
    return dst

# -------------------------------------------------------------------------------------

pulse0 = 80128
pulse1 = 92504

fname = 'dst_bolo.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in range(pulse0, pulse1+1):
    try:
        ipla, ipla_t = get_data(pulse, 'magn', 'ipla')
        dst = get_dst(ipla, ipla_t)
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
