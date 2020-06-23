import utils

from astropy.stats import gaussian_fwhm_to_sigma
from object_detector import image_segmentation

import progressbar

# config variables
npixels = 3
r = 3.  # approximate isophotal extent
sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
threshold_value = 1.5
train_data_path = '../data/spotGEO/train_anno.json'

initial_image = 1
last_image = 1

# tools that contain auxiliary functions
tools = utils.Utils(train_data_path)

bar = progressbar.ProgressBar(maxval=((last_image+1) - initial_image)*5, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for image in range(initial_image, (last_image+1)):
    for frame in range(1,6):

        # get initial data for the image and frame given
        [data, geo_objects] = tools.get_initial_data(image, frame)

        # object detector => detect all objects that fits the parameters (bright, size, shape, ...)
        apertures = image_segmentation(data, npixels, r, sigma, threshold_value)

        # geos_filter => from all detected objects, select just the geos (using a convolutional neural network)

        # trajectory filter => connect all geos from all frames of the same sequence

        # center finder => get the center coords of the detected geos

        # write solution json => write the json with the requested format

        # plot data
        tools.generate_plots(data, geo_objects, apertures, npixels, threshold_value, image, frame)

        # it show a progressbar to know the progress
        bar.update((image-1)*5 + (frame-1))

bar.finish()