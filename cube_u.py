import h5py
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from astropy.io import fits


f = h5py.File('archive_u.hdf5', 'a')
times = np.loadtxt('times_u.txt')
altitude = np.loadtxt('altitude_u.txt')
airmass = np.loadtxt('airmass_u.txt')

dset = f['images']
# dset_u = u['images']
background = np.median(dset[:], axis=(0, 1))
target = dset[145:165, 50:70, :]
comparison1 = dset[160:180, 80:100, :]
comparison2 = dset[40:70, 240:260, :]
comparison3 = dset[190:210, 180:200, :]
comparison4 = dset[120:140, 50:70, :]
comparison5 = dset[250:270, 35:55, :]

# plt.imshow(np.log(np.sum(dset[:], axis=2)), origin='lower')
# plt.scatter([60, 90, 250, 200, 60, 45], [155, 170, 55, 190, 130, 260], color='w')
# plt.show()

# plt.imshow(np.sum(target, axis=2))
# plt.show()

# # comparison2 = dset[165:205, 260:300, :]
# target_bg = np.median(target, axis=(0, 1))
# comparison1_bg = np.median(comparison1, axis=(0, 1))
# comparison2_bg = np.median(comparison2, axis=(0, 1))
# comparison3_bg = np.median(comparison3, axis=(0, 1))
# comparison4_bg = np.median(comparison4, axis=(0, 1))
# comparison5_bg = np.median(comparison5, axis=(0, 1))

target_flux = np.sum(target, axis=(0, 1)) #- target_bg
comp_flux1 = np.sum(comparison1, axis=(0, 1)) #- comparison1_bg
comp_flux2 = np.sum(comparison2, axis=(0, 1)) #- comparison2_bg
comp_flux3 = np.sum(comparison3, axis=(0, 1)) #- comparison2_bg
comp_flux4 = np.sum(comparison4, axis=(0, 1)) #- comparison4_bg
comp_flux5 = np.sum(comparison5, axis=(0, 1)) #- comparison5_bg

# print(np.shape(target_bg))
# plt.imshow(np.log(np.sum(comparison1_bg, axis=-1)))
# plt.show()
#
# from astroplan import FixedTarget, Observer
# from astropy.time import Time
# apo = Observer.at_site("APO")
# aa = apo.altaz(Time(times, format='jd'), FixedTarget.from_name('KIC9652680'))
# airmass = aa.secz.value
# altitude = aa.alt.rad
# np.savetxt('airmass_u.txt', airmass)
# np.savetxt('altitude_u.txt', altitude)


mask_outliers = np.ones_like(target_flux).astype(bool)

X = np.vstack([comp_flux1, comp_flux3, comp_flux5, #comp_flux2, #comp_flux3, #comp_flux4, #comp_flux5,
               1-airmass, altitude, background]).T

c = np.linalg.lstsq(X[mask_outliers], target_flux[mask_outliers])[0]
comparison = X @ c
print(c)

lc = target_flux/comparison

fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
ax[0].plot(times, lc, '.')
ax[0].plot(times[~mask_outliers], lc[~mask_outliers], '.')
ax[1].plot(times[mask_outliers], target_flux[mask_outliers], '.', label='target')
ax[1].plot(times[mask_outliers], comparison[mask_outliers], '.', label='comp')
ax[1].legend()
for i, comp in enumerate([comp_flux1, comp_flux2, comp_flux3, comp_flux4, comp_flux5]):
    ax[2].plot(times[mask_outliers], comp[mask_outliers], '.', label='{0}'.format(i+1))
ax[2].legend()
# ax[4].plot(times[mask_outliers], comp_flux3[mask_outliers], '.')
# ax[2].plot(times, airmass, '.')
# plt.plot(times, comp_flux1, '.')
# plt.plot(times, comp_flux2, '.')

np.savetxt('lc_u.txt', np.vstack([times, lc]).T)

plt.show()