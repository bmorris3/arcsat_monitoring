import h5py
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from astropy.io import fits


f = h5py.File('archive.hdf5', 'a')
dset = f['images']
target = dset[150:190, 150:190, :]
#comparison = dset[170:200, 185:220, :]
comparison = dset[165:205, 260:300, :]
target_bg = np.median(target, axis=(0, 1))
comparison_bg = np.median(comparison, axis=(0, 1))

target_flux = np.sum(target, axis=(0, 1)) - target_bg
comp_flux = np.sum(comparison, axis=(0, 1)) - comparison_bg

times = np.loadtxt('times.txt')
plt.plot(times, target_flux/comp_flux, '.')
plt.show()