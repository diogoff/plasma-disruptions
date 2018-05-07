from __future__ import print_function

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

# -------------------------------------------------------------------------------------

def get_dst(ipla, ipla_t):
    x0 = ipla_t[:-1]
    x1 = ipla_t[1:]
    y0 = ipla[:-1]
    y1 = ipla[1:]
    grad = (y1-y0)/(x1-x0)
    grad_t = (x0+x1)/2.
    dst = None
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

fname = 'dst.txt'
print('Writing:', fname)
f = open(fname, 'w')

for pulse in range(pulse0, pulse1+1):
    try:
        ipla, ipla_t = get_data(pulse, 'magn', 'ipla')
    except ValueError:
        continue
    
    dst = get_dst(ipla, ipla_t)
    
    if dst != None:
        print('%d %.4f' % (pulse, dst))
        f.write('%d %.4f\n' % (pulse, dst))
        f.flush()

f.close()
