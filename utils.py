import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import json

class Utils():

    def __init__(self, train_data_path=''):

        if train_data_path != '':
            self.geo_objects_info = []
            with open(train_data_path) as json_data:
                self.geo_objects_info = json.load(json_data)


    def generate_plots(self, data, geo_objects, apertures, npixels, threshold_value, kernel_x_size, kernel_y_size, image, frame):

        norm = ImageNormalize(stretch=SqrtStretch())

        #Plot original in ax1 and processed in ax2
        _, ax = plt.subplots(1, 1, figsize=(10, 12.5))

        ax.imshow(data, cmap='Greys_r', norm=norm)
        ax.set_title('segmentations plot => n_pixels: '+str(npixels)+' threshold value: '+str(threshold_value))

        # plot real objects with caracter '+'
        for geo_obj in geo_objects:
            ax.plot(geo_obj[0], geo_obj[1], ls='none', color='red', marker='+', ms=10, lw=1.5)

        for aperture in apertures:
            aperture.plot(axes=ax, color='blue', lw=1.5)

        plt.title('image: '+str(image)+' frame: '+str(frame)+' => '+ str(len(apertures)) + ' objects detected (only ' + str(len(geo_objects)) + ' are real)')
        plt.savefig('./results/segmentations plot image: '+str(image)+' frame: '+str(frame)+' => threshold value: '+str(threshold_value)+' X-Y kernel size:'+str(kernel_x_size)+'x'+str(kernel_y_size)+'.png')

    def get_initial_data(self, dataset, image, frame):

        position_json = (image-1)*5 + (frame-1)
        data = mpimg.imread('../data/spotGEO/'+dataset+'/'+str(image)+'/'+str(frame)+'.png')
        
        if dataset == 'test':
            return data
        
        geo_objects = self.geo_objects_info[position_json]['object_coords']

        return [data, geo_objects]
        
