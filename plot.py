import h5py
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from astropy.io import fits


f = h5py.File('archive.hdf5', 'a')
dset = f['images']

composite = np.sum(f['images'][:], axis=2)

# composite = composite.copy() - np.min(composite) + 1

# plt.hist(composite.ravel(), log=True)
plt.imshow(np.log(composite), vmin=13, vmax=15, origin='lower')
plt.show()