from __future__ import print_function

import sys
import h5py
import tqdm
import numpy as np

# ----------------------------------------------------------------------

if len(sys.argv) < 3:
    print('Usage: %s prd_limit ttd_limit' % sys.argv[0])
    exit()
    
# ----------------------------------------------------------------------

prd_limit = float(sys.argv[1])
ttd_limit = float(sys.argv[2])

print('prd_limit: %.2f' % prd_limit)
print('ttd_limit: %.2f' % ttd_limit)

# ----------------------------------------------------------------------

fname = 'dst_pred.hdf'
print('Reading:', fname)
f = h5py.File(fname, 'r')

print('pulses:', len(f))

# ----------------------------------------------------------------------

TP = 0
TN = 0
FP = 0
FN = 0

success_rate = []
warning_time = []

for pulse in tqdm.tqdm(f):

    dst = f[pulse]['dst'][0]
    prd = f[pulse]['prd'][:]
    ttd = f[pulse]['ttd'][:]
    prd_t = f[pulse]['prd_t'][:]
    ttd_t = f[pulse]['ttd_t'][:]

    alarm = 0.
    for i in range(prd_t.shape[0]):
        if (dst > 0.) and (prd_t[i] >= dst):
            break
        if (prd[i] >= prd_limit) and (ttd[i] <= ttd_limit):
            alarm = prd_t[i]
            break
    if dst > 0.:
        if alarm > 0.:
            TP += 1
            success_rate.append(1.)
            warning_time.append(dst - alarm)
        else:
            FN += 1
            success_rate.append(0.)
    else:
        if alarm > 0.:
            FP += 1
        else:
            TN += 1

print('TP: %3d (%5.2f%%)' % (TP, float(TP) / float(len(f)) * 100.))
print('TN: %3d (%5.2f%%)' % (TN, float(TN) / float(len(f)) * 100.))
print('FP: %3d (%5.2f%%)' % (FP, float(FP) / float(len(f)) * 100.))
print('FN: %3d (%5.2f%%)' % (FN, float(FN) / float(len(f)) * 100.))

# ----------------------------------------------------------------------

precision = float(TP) / float(TP + FP)
recall = float(TP) / float(TP + FN)
f_measure = 2. * precision * recall / (precision + recall)

print('precision: %6.2f' % (precision*100.))
print('recall:    %6.2f' % (recall*100.))
print('f_measure: %6.2f' % (f_measure*100.))

success_rate = np.mean(success_rate)
warning_time = np.mean(warning_time)

print('success_rate: %6.2f%%' % (success_rate*100.))
print('warning_time: %5.1f ms' % (warning_time*1000.))

# ----------------------------------------------------------------------

f.close()
