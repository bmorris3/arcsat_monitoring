
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.utils.console import ProgressBar

from astroscrappy import detect_cosmics

paths = sorted(glob('/Users/bmmorris/data/arcsat/*/KIC96*sdss_g*.fits'))#[:100]

image_shape = fits.getdata(paths[0]).shape

f = h5py.File('archive.hdf5', 'w')

if 'images' not in f:
    dset = f.create_dataset("images", shape=(image_shape[0], image_shape[1],
                                             len(paths)))
else:
    dset = f['images']

brightest_star_coords_init = np.array([186, 280])

master_flat_path = 'outputs/masterflat_20160708.fits'
master_dark_path = 'outputs/masterdark_20160708.fits'

flat = fits.getdata(master_flat_path)
dark = fits.getdata(master_dark_path)

from skimage.feature import peak_local_max

mid = image_shape[0]//2

with ProgressBar(len(paths)) as bar:
    for i, path in enumerate(paths):

        raw_image = fits.getdata(path) / flat
        # mask, cleaned_image = detect_cosmics(raw_image)
        #cleaned_image = raw_image
        # smoothed_image = gaussian_filter(cleaned_image, 10)

        coordinates = peak_local_max(raw_image, min_distance=10,
                                     num_peaks=1, exclude_border=10)
        y_mean = int(coordinates[:, 1].mean())
        x_mean = int(coordinates[:, 0].mean())

        # y, x = np.unravel_index(np.argmax(smoothed_image), smoothed_image.shape)

        print(mid - y_mean, mid - x_mean)
        firstroll = np.roll(raw_image, mid - y_mean,#x-brightest_star_coords_init[1],
                            axis=1)
        rolled_image = np.roll(firstroll, mid - x_mean, #y-brightest_star_coords_init[0],
                               axis=0)

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(np.log(raw_image), origin='lower')
        # ax[1].imshow(np.log(rolled_image), origin='lower')
        # ax[0].scatter(coordinates[:, 1], coordinates[:, 0], color='w')
        # plt.show()

        # if i == 20:
        #     plt.imshow(np.sum(dset[:, :, :], axis=2))
        #     plt.show()

        # if i < 100:
        #
        #     plt.imshow(np.log(cleaned_image), origin='lower')
        #     plt.scatter(coordinates[:, 1], coordinates[:, 0], color='w')
        #     # plt.scatter(x, y, color='r')
        #     plt.show()
        dset[:, :, i] = rolled_image

        bar.update()