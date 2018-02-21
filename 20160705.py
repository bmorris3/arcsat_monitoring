import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from toolkit import (generate_master_flat_and_dark, photometry, transit_model_b,
                     PhotometryResults, PCA_light_curve, params_b)

# Image paths

image_paths = sorted(glob('/Users/bmmorris/data/arcsat/20160705/KIC*sdss_g*.fits'))#[:20]
dark_30s_paths = glob('/Users/bmmorris/data/arcsat/20160705/Dark*.fits')
night_flat_paths = glob('/Users/bmmorris/data/arcsat/20160705/domeflat_sdss_g*.fits')
master_flat_path = 'outputs/masterflat_20160705.fits'
master_dark_path = 'outputs/masterdark_20160705.fits'

# Photometry settings
aperture_radii = [8]#np.arange(4, 15)
centroid_stamp_half_width = 8
psf_stddev_init = 2
aperture_annulus_radius = 10
transit_parameters = params_b
star_positions = np.loadtxt('20160705_g.txt')

brightest_star_coords_init = np.array([186, 280])

output_path = 'outputs/20160705.npz'
force_recompute_photometry = True

# Calculate master dark/flat:
if not os.path.exists(master_dark_path) or not os.path.exists(master_flat_path):
    print('Calculating master flat:')
    generate_master_flat_and_dark(night_flat_paths, dark_30s_paths,
                                  master_flat_path, master_dark_path)

# Do photometry:

if not os.path.exists(output_path) or force_recompute_photometry:
    print('Calculating photometry:')
    phot_results = photometry(image_paths, master_dark_path, master_flat_path,
                              star_positions, aperture_radii, centroid_stamp_half_width,
                              psf_stddev_init, aperture_annulus_radius,
                              output_path, brightest_star_coords_init)

else:
    phot_results = PhotometryResults.load(output_path)

# print('Calculating PCA...')


target = phot_results.fluxes[:, 0, 0]
mean_comparison = phot_results.fluxes[:, 1, 0]
light_curve = target/mean_comparison
plt.plot(phot_results.times, light_curve/np.median(light_curve), '.')

np.savetxt('outputs/20160705.txt',
           np.vstack([phot_results.times, light_curve]).T)

plt.legend()
plt.show()



# light_curve = PCA_light_curve(phot_results, transit_parameters, plots=False,
#                               plot_validation=False, buffer_time=1*u.min,
#                               validation_duration_fraction=0.8,
#                               validation_time=1.2, outlier_rejection=True)
#
# plt.figure()
# plt.plot(phot_results.times, light_curve, 'k.')
# # plt.plot(phot_results.times, transit_model_b(phot_results.times), 'r')
# plt.xlabel('Time [JD]')
# plt.ylabel('Flux')
# plt.show()