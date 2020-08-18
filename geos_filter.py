import numpy as np
import copy
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator  

class GEOs_filter():

    def __init__(self, pixels_from_center):

        self.pixels_from_center = pixels_from_center

        self.x_dimension = (self.pixels_from_center*2)+1
        self.y_dimension = (self.pixels_from_center*2)+1

    def generate_train_dataset(self, data, apertures, geo_objects, image_number, frame):

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

                image = self.get_image_from_coords(data, image_number, frame, int(sample.positions[0]), int(sample.positions[1]), False)
                if len(image) != 0:
                    negative_samples.append(image)

        positive_samples = []
        # generate images for positives samples
        for sample in geo_objects:
            image = self.get_image_from_coords(data, image_number, frame, int(sample[0]), int(sample[1]), True)
            if len(image) != 0:

                positive_samples.append(image)

                enhanced_positive_sample = self.enhance_positive_sample(image, image_number, frame, int(sample[0]), int(sample[1]))
                positive_samples += enhanced_positive_sample

        self.train(positive_samples, negative_samples)

    def extract_real_geos(self, apertures):

        return 0

    def get_image_from_coords(self, data, image_number, frame, x, y, is_positive_sample):

        image = []

        for new_x in range(x-self.pixels_from_center, x+1+self.pixels_from_center):
            row = []
            for new_y in range(y-self.pixels_from_center, y+1+self.pixels_from_center):
                if new_x >= 0 and new_x < len(data[0]) and new_y >= 0 and new_y < len(data):
                    row.append(data[new_y][new_x])
                else:
                    return []
            
            image.append(row)

        mat = np.reshape(image, (self.x_dimension, self.y_dimension))

        # Creates PIL image
        img = Image.fromarray(np.uint8(mat*255) , 'L')

        if is_positive_sample:
            img.save('./dataset/positive/'+str(image_number)+'_'+str(frame)+'_'+str(x)+'_'+str(y)+'.png')
        
        else:
            img.save('./dataset/negative/'+str(image_number)+'_'+str(frame)+'_'+str(x)+'_'+str(y)+'.png')


        return image

    # this method generate multiple samples from one positive samples
    # the transformations applied are: rotations (multiple angles), mirroring (x and y axis), 
    def enhance_positive_sample(self, image, image_number, frame, x, y):

        enhanced_samples = []

        # 90 degrees rotation
        for index_rotation in range (1, 4):

            rotated_image = np.rot90(image, k=index_rotation, axes=(1,0))
            enhanced_samples.append(rotated_image)
            
            mat = np.reshape(rotated_image, (self.x_dimension, self.y_dimension))

            # Creates PIL image
            img = Image.fromarray(np.uint8(mat*255) , 'L')
            img.save('./dataset/positive/'+str(image_number)+'_'+str(frame)+'_'+str(x)+'_'+str(y)+'_rotated_'+str((index_rotation*90))+'.png')

        # flip (mirroring)
        #vertical
        flipped_image = copy.deepcopy(image)
        flipped_image.reverse()

        enhanced_samples.append(flipped_image)

        mat = np.reshape(flipped_image, (self.x_dimension, self.y_dimension))

        # Creates PIL image
        img = Image.fromarray(np.uint8(mat*255) , 'L')
        img.save('./dataset/positive/'+str(image_number)+'_'+str(frame)+'_'+str(x)+'_'+str(y)+'_flipped_vertical.png')

        #horizontal
        flipped_image = copy.deepcopy(image)

        for row in flipped_image:
            row.reverse()

        enhanced_samples.append(flipped_image)

        mat = np.reshape(flipped_image, (self.x_dimension, self.y_dimension))

        # Creates PIL image
        img = Image.fromarray(np.uint8(mat*255) , 'L')
        img.save('./dataset/positive/'+str(image_number)+'_'+str(frame)+'_'+str(x)+'_'+str(y)+'_flipped_horizontal.png')

        return enhanced_samples

    def train(self, positive_samples, negative_samples):

        # Actual (small dataset): https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        # General CNN with TF: https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa
        # Binary Classifier Keras+TF: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
        # Basic approach Keras: https://www.geeksforgeeks.org/python-image-classification-using-keras/
        # Repositories for binary classificators: https://github.com/topics/binary-image-classification
        # LeNet: https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
        # Keras classifier: https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
        # Keras+TF official documentation: https://www.tensorflow.org/tutorials/keras/classification?hl=es-419

        # generate model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(self.x_dimension, self.y_dimension, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        batch_size = 16

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                'data/train',  # this is the target directory
                target_size=(self.x_dimension, self.y_dimension),  # all images will be resized
                batch_size=batch_size,
                class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
                'data/validation',
                target_size=(self.x_dimension, self.y_dimension),
                batch_size=batch_size,
                class_mode='binary')

        model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=validation_generator,   
            validation_steps=800 // batch_size)

        model.save_weights('detect_geos_model.h5')  # always save your weights after training or during training

        return 0

