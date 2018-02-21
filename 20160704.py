import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from toolkit import (generate_master_flat_and_dark, photometry,
                     PhotometryResults, PCA_light_curve, params_b)

# Image paths
# image_paths = sorted(glob('/Users/bmmorris/data/arcsat/20160704/KIC*sdss_g*.fits'))
# dark_30s_paths = glob('/Users/bmmorris/data/arcsat/20160704/Dark*.fits')
# night_flat_paths = glob('/Users/bmmorris/data/arcsat/20160704/domeflat_sdss_g*.fits')
# master_flat_path = 'outputs/masterflat_20160704.fits'
# master_dark_path = 'outputs/masterdark_20160704.fits'

image_paths = sorted(glob('/Users/bmmorris/data/arcsat/20160704/KIC*sdss_u*.fits'))
dark_30s_paths = glob('/Users/bmmorris/data/arcsat/20160704/Dark*.fits')
night_flat_paths = glob('/Users/bmmorris/data/arcsat/20160704/domeflat_sdss_u*.fits')
master_flat_path = 'outputs/masterflat_20160704.fits'
master_dark_path = 'outputs/masterdark_20160704.fits'

# Photometry settings
target_centroid = [465, 456]
comparison_flux_threshold = 0.05
aperture_radii = np.arange(4, 15)
centroid_stamp_half_width = 8
psf_stddev_init = 2
aperture_annulus_radius = 10
transit_parameters = params_b
star_positions = np.loadtxt('/Users/bmmorris/Desktop/20160704_u.txt')
#star_positions = np.loadtxt('/Users/bmmorris/Desktop/night1.txt')#np.loadtxt('toolkit/data/arcsat.txt')
#star_positions = [[y, x] for x, y in star_positions]

brightest_star_coords_init = np.array([186, 282])

output_path = 'outputs/20160704.npz'
force_recompute_photometry = False#True

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


for i in range(phot_results.fluxes.shape[2]):
    target = phot_results.fluxes[:, 0, i]
    mean_comparison = np.mean(phot_results.fluxes[:, 1:, i], axis=1)
    light_curve = target/mean_comparison
    plt.plot(phot_results.times, light_curve/np.median(light_curve), '.', label=i)
plt.legend()
plt.show()


# print('Calculating PCA...')

# plt.plot(phot_results.times, phot_results.fluxes[:, 0, :])
# plt.show()
#
# light_curve = PCA_light_curve(phot_results, transit_parameters, plots=True,
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