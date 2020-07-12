import numpy as np
from PIL import Image

class GEOs_filter():

    def __init__(self, pixels_from_center):

        self.pixels_from_center = pixels_from_center

    def train(self, data, apertures, geo_objects):

        negative_samples = []
        # generate images for negative samples
        for sample in apertures:

            is_positive_sample = False
            # check if it is a possible sample
            for positive_sample in geo_objects:
                if abs(positive_sample[0]-sample.positions[0]) < self.pixels_from_center and abs(positive_sample[1] - sample.positions[1]) < self.pixels_from_center:
                    
                    print('Image with x:', sample.positions[0], 'and y:', sample.positions[1], 'discarded for positive proximity!!')
                    is_positive_sample = True
                    break

            if not is_positive_sample:

                image = self.get_image_from_coords(data, int(sample.positions[0]), int(sample.positions[1]))
                if len(image) != 0:
                    negative_samples.append(image)

        positive_samples = []
        # generate images for positives samples
        for sample in geo_objects:
            image = self.get_image_from_coords(data, int(sample[0]), int(sample[1]))
            if len(image) != 0:
                positive_samples.append(image)

        print('training')

    def extract_real_geos(self, apertures):

        return 0

    def get_image_from_coords(self, data, x, y):

        image = []

        for new_x in range(x-self.pixels_from_center, x+1+self.pixels_from_center):
            row = []
            for new_y in range(y-self.pixels_from_center, y+1+self.pixels_from_center):
                if new_x >= 0 and new_x < len(data[0]) and new_y >= 0 and new_y < len(data):
                    row.append(data[new_y][new_x])
                else:
                    return []
            
            image.append(row)

        mat = np.reshape(image,((self.pixels_from_center*2)+1,(self.pixels_from_center*2)+1))
        # Creates PIL image
        img = Image.fromarray(np.uint8(mat*255) , 'L')
        img.save('./results/'+str(x)+'_'+str(y)+'.png')
        print('<i>', x,' ', y, 'image generated!')

        return image

