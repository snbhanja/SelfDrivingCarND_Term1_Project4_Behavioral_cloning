import csv
import cv2
import data_augmentation_functions
import numpy as np
from nvidia_model import model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Class for the image processing and training pieline.
# The parameters are:
#   1. model : The keras model 
#   2. base_path : The path where the images folder 'IMG' and driving_log.csv present.
class ImagePipeline:
    def __init__(self, model=None, base_path='', epochs=2):
        # Store lines read in from driving_log.csv
        self.data = []

        # The model present in nvidia_model.py file.
        self.model = model

        # The number of epochs for which the model will run.
        self.epochs = epochs

        # batch size
        self.batch_size = 128
        
        # self.training_samples and self.validation_samples will store the result of running 
        # train_test_split on data stored in self.data
        self.training_samples = []
        self.validation_samples = []

        # The steering angle adjustment for left and right camera images used during data augmentation. 
        self.correction_factor = 0.2

        # The path where the images folder 'IMG' and driving_log.csv present. 
        self.base_path = base_path

        # The path to training images.
        self.image_path = self.base_path + '/IMG/'

        # The path to driving_log.csv file.
        self.driving_log_path = self.base_path + '/driving_log.csv'

    def import_data(self):
        # This function open the driving log CSV file and reads each row into self.data
        # The file contains absolute paths to the center, left and right camera images as well as the steering angle.
        with open(self.driving_log_path) as csvfile:
            reader = csv.reader(csvfile)
            # Skip the column names row
            next(reader)

            for line in reader:
                self.data.append(line)

        return None

    def process_batch(self, batch_sample):
        # This function does the image and steering angle data augmentation.
        # The data_augmentation_functions.py file have the image augmentation functions.
        # Parameters: 
        #     batch_sample: is a list containing the paths to the center, left and right images
    
        # Convert steering angle to float.
        steering_angle = np.float32(batch_sample[3])
        
        # empty lists defined to hold the results of the data augmentation.
        images, steering_angles = [], []

        # There are three image paths in each row. The below for loop runs for each image path
        for image_path_index in range(3):
            # Get the image name only splitting path by '/' and taking the last element.
            image_name = batch_sample[image_path_index].split('/')[-1]

            # cv2.imread function from OpenCV is used to read in an image. 
            image = cv2.imread(self.image_path + image_name)

            # Since OpenCV reads in images in the BGR color space and 
            #  because drive.py will feed RGB images to our model, we convert the image variable 
            #  containing a numpy array representation of the image into RGB using data_augmentation_functions.bgr2rgb function
            #  so that our model is trained on RGB images.
            rgb_image = data_augmentation_functions.bgr2rgb(image)
                
            # rgb_image is then cropped to remove the sky and the hood and resized to (70, 160) 
            resized = data_augmentation_functions.crop_and_resize(rgb_image)

            # Each image is appended to the images list.
            images.append(resized)

            # Apply correction factor 0.2 to left image. i.e. add 0.2 to left image steering angle.
            if image_path_index == 1:
                steering_angles.append(steering_angle + self.correction_factor)
            # Apply correction factor 0.2 to right image. i.e. substract 0.2 from the right image steering angle.
            elif image_path_index == 2:
                steering_angles.append(steering_angle - self.correction_factor)
            else:
                steering_angles.append(steering_angle)

            # for every center image, image is flipped horizontally, added to the images list 
            # and the opposite of the steering angle is added to steering_angles.
            if image_path_index == 0:
                flipped_center_image = data_augmentation_functions.flipimg(resized)
                images.append(flipped_center_image)
                steering_angles.append(-steering_angle)

        return images, steering_angles

    def data_generator(self, samples, batch_size=128):
        # The function to generate samples of data which will be passed to model.
        # Parameters:
        #     samples: is the data from which to create the generator.
        #              The data will be coming from self.training_samples and self.validation_samples.
        #     batch_size : is the number of samples desired in a single batch.
        #                  Here a batch of 128 samples generated.		
        num_samples = len(samples)

        # create an infinite loop so that we can call our generator as many times as we wish.
        while True:
            shuffle(samples)

            # The first loop iterates through samples using batch_size as the step size
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images, steering_angles = [], []
            
                #  The subsequent loop iterates through each sample in a given batch.
                # Two lists — images and steering_angles are created to store the result of our data augmentation process.
                for batch_sample in batch_samples:
                    # self.process_batch will be called to generate augmented images and steering angles
                    augmented_images, augmented_angles = self.process_batch(batch_sample)

                    # The augmented images and steering angles are then added to the images and steering_angles lists respectively
                    images.extend(augmented_images)
                    steering_angles.extend(augmented_angles)

                # images and steering_angles converted to numpy arrays and assigned to X_train and y_train variables. 
                # X_train and y_train will then be used as the starting point when data_generator is called subsequently
                X_train, y_train = np.array(images), np.array(steering_angles)

                # The next time the data_generator is called, it will not start from the scratch, 
                # but will resume where yield left off, keeping the generator’s variables intact.
                yield shuffle(X_train, y_train)

    def split_data(self):
        # the data is then split into a training and validation sets in ratios 0.8 and 0.2
        # and assigned to the instance variables self.training_samples and self.validation_samples
        train, validation = train_test_split(self.data, test_size=0.2)
        self.training_samples, self.validation_samples = train, validation

        return None

    # Train set generator.	
    def train_generator(self, batch_size=128):
        return self.data_generator(samples=self.training_samples, batch_size=batch_size)

    # Validation set generator.
    def validation_generator(self, batch_size=128):
        return self.data_generator(samples=self.validation_samples, batch_size=batch_size)

    def run(self):
        # This function finally runs the pipeline.

        # First split the data into training and vaidation set.
        self.split_data()

        # Run Keras's method model.fit_generator to start the training.
        # steps_per_epoch is telling Keras how many batches to create for each epoch.
        self.model.fit_generator(generator=self.train_generator(),
                                 validation_data=self.validation_generator(),
                                 epochs=self.epochs,
                                 steps_per_epoch=len(self.training_samples) * 2,
                                 validation_steps=len(self.validation_samples),
                                 verbose=1)
        self.model.save('model.h5')
        print("Model training finished...")
        

def main():
    #Pass the parameters to ImagePipeline function.
    imagepipeline = ImagePipeline(model=model(), base_path="../data", epochs=2)
    # Feed driving log data into the imagepipeline
    imagepipeline.import_data()
    # Start training
    imagepipeline.run()

if __name__ == '__main__':
    main()