# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./writeup_images/nvidia_model.png "Original NVIDIA paper"
[image2]: ./writeup_images/model_summary.JPG "Model summary"
[image3]: ./writeup_images/model_plot.png "Model plot"
[image4]: ./writeup_images/raw_center_1.jpg "center image"
[image5]: ./writeup_images/raw_left_1.jpg "left image"
[image6]: ./writeup_images/raw_right_1.jpg "right image"
[image7]: ./writeup_images/rgb_image.png "rgb image"
[image8]: ./writeup_images/crop_image.png "cropped image"
[image9]: ./writeup_images/crop_and_resize.png "cropped and resized"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* data_augmentation_functions.py containing data augmentation functions
* nvidia_model.py containing the nvidia model architecture.
* model.py containing the script to create and train the model. Above two python scripts used here.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* out_image.mp4 The output video of auonomous driving in track 1.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The code for model architecture present in nvidia_model.py file.

The Nvidia model is used because of its simplicity and demonstrated ability to perform well on self-driving car tasks. 
Please follow the below research paper,
https://arxiv.org/pdf/1604.07316v1.pdf

The Nvidia architecture uses images of with a shape of (66, 200, 3), I have changed the input_shape to be (70, 160, 3).
The architecture from NVIDIA paper as below,

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

I have tried with Keras BatchNormalization for reducing overfit, Then model was tested by running it through the simulator, but the vehicle could not stay on the track after crossing the Bridge. The network architecture present in 'nvidia_model_batchnorm.py' and the trained model file is 'model_batchnorm.h5'.

Then, I removed the BatchNormalization and reran the training and tested the model passing it to drive.py.
This time the car completed the lap without deviating from the track.
The model architecture present in 'nvidia_batchnorm.py' and 'model.h5' is the model file created.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 4).
The loss function used is 'mse', since it is a regression problem.
Batch size of 128 used.

#### 4. Appropriate training data

For training I used the data provided by udacity.
The shell script 'get_training_data.sh' fetch the data from dropbox and extract into 'data' folder outside 'workspace' directory.
The data has a combination of center lane driving, recovering from the left and right sides of the road.
Also, it has the 'driving_log.csv' file which has the absolute paths of all the theree images and the steering angles data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The strategy for deriving a good model was to use the Nvidia architecture since it has been proven to be very successful in self-driving car tasks. The architecture was recommended in the lessons and it's adapted for this use case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in 80% and 20% ratio. Since I was using data augmentation techniques, the mean squared error was low both on the training and validation steps.

I had some problems in 'fit_generator' function for setting the parameter steps_per_epoch.
steps_per_epoch is telling Keras how many batches to create for each epoch. After some experimentation I set it to two times
length of training sample.

As, discussed earlier, I tried with BatchNormalization, but the car went off the track after bridge.

The final step was to run the simulator to see how well the car was driving around track one. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.<br/>
![alt text][image2]<br/><br/>

Here is a visualization of the architecture.<br/>
![alt text][image3]
<br/>
#### 3. Creation of the Training Set & Training Process

To create the training data, I used the Udacity sample data. For each image, normalization applied before the image was fed into the network. In my case, a training sample consisted of four images:

* Center camera image
* Horizontally flipped center camera image
* Left camera image
* Right camera image

The sample raw images as below,

Center camera image: <br/>
![alt text][image4] <br/><br/>

Left camera image: <br/>
![alt text][image5] <br/><br/>

Right camera image: <br/>
![alt text][image6] <br/><br/>

The BGR images converted to RGB image, below is the sample RGB image,<br/>
![alt text][image7]<br/><br/>

Cropped image as below,<br/>
![alt text][image8]<br/><br/>

After resizing the cropped image to 160x70 pixels ,<br/>
![alt text][image9]<br/><br/>

The steering angle for left and right images as below,<br/>
Applied correction factor 0.2 to left image. i.e. add 0.2 to left image steering angle.<br/>
Applied correction factor 0.2 to right image. i.e. substract 0.2 from the right image steering angle.<br/><br/>

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Finally, the below link has the autonomous drive video of track 1.

https://www.youtube.com/watch?v=alH2p_MzHlA&feature=youtu.be

Inside workspace, the output video file is "out_image.mp4".

#### Future work:
To collect data for track 2 and test the model performance.
