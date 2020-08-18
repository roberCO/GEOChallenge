import sys

import geos_filter
import utils

from astropy.stats import gaussian_fwhm_to_sigma

from object_detector import image_segmentation

import progressbar

mode = ''
if len(sys.argv) == 2:

    print('<i> Trainig mode')
    mode = 'training'
    train_data_path = '../data/spotGEO/train_anno.json'

    # tools that contain auxiliary functions
    tools = utils.Utils(train_data_path)

elif len(sys.argv) == 1:

    print('<1> Detecting GEOs mode')
    mode = 'execution'
    train_data_path = '../data/spotGEO/test_anno.json'

    tools = utils.Utils()

else:

    print ("<*> ERROR: Wrong number of parameters - Usage: python main.py [execution_mode]")
    print ("<!> Example: python main.py train")
    sys.exit(0)

initial_image = 1
last_image = 1

# config variables
npixels = 3
r = 3.  # approximate isophotal extent
sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
kernel_x_size = 2
kernel_y_size = 2
threshold_value = 1.5
max_area = 80
min_area = 15
threshold_roundity = 0.5
pixels_from_center = 5

geos_filter = geos_filter.GEOs_filter(pixels_from_center)

#bar = progressbar.ProgressBar(maxval=((last_image+1) - initial_image)*5, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#bar.start()
for image in range(initial_image, (last_image+1)):
    for frame in range(1,2):

        if mode == 'training':
            dataset = 'train'

            # get initial data for the image and frame given
            [data, geo_objects] = tools.get_initial_data(dataset, image, frame)
        
        else:
            dataset = 'test'

            # get initial data for the image and frame given
            data = tools.get_initial_data(dataset, image, frame)
        
        # object detector => detect all objects that fits the parameters (bright, size, shape, ...)
        apertures = image_segmentation(data, npixels, r, sigma, threshold_value, kernel_x_size, kernel_y_size, min_area, max_area, threshold_roundity)

        if mode == 'training':
            geos_filter.generate_train_dataset(data, apertures, geo_objects, image, frame)

        else:
            # geos_filter => from all detected objects, select just the geos (using a convolutional neural network)
            filtered_geos = geos_filter.extract_real_geos(apertures)

        # trajectory filter => connect all geos from all frames of the same sequence

        # center finder => get the center coords of the detected geos

        # write solution json => write the json with the requested format

        # plot data
        tools.generate_plots(data, geo_objects, apertures, npixels, threshold_value, kernel_x_size, kernel_y_size, image, frame)

        # it show a progressbar to know the progress
        #bar.update((image-1)*5 + (frame-1))

#bar.finish()