# **Traffic Sign Recognition** 

## Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images_from_web_cropped/1-kids-crossing-sign.jpg "Traffic Sign 1"
[image5]: ./images_from_web_cropped/2-no-entry-sign.jpg "Traffic Sign 2"
[image6]: ./images_from_web_cropped/3-speed-limit-60-sign.jpeg "Traffic Sign 3"
[image7]: ./images_from_web_cropped/4-stop-sign.jpg "Traffic Sign 4"
[image8]: ./images_from_web_cropped/5-priority-road-sign.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/peterkinmond/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing summing up the number of each type of traffic sign in the training set. Interestingly the number of each sign type varies quite a bit, ranging from ~200 to ~2000.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I preprocessed the image data by normalizing all the images. This helped the model by making all the data have equal variance. I considered converting all images to greyscale but was able to get a high matching percentage without doing that.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| #1 Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| #2 Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16       									|
| RELU					|   |
| Max pooling			| 2x2 stride, outputs 5x5x16 |
| Flatten		| Outputs 400        									|
| Fully connected		| outputs 120        									|
| RELU		| 						|
| Dropout		| keep probability: 0.7 						|
| Fully connected		| outputs 84									|
| RELU		| 						|
| Dropout		| keep probability: 0.7 						|
| Fully connected		| outputs 43 |	


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I tried many different combinations of tuning the hyperparameters. I ended up with the following values for each hyperparameter:

* Batch size: 128
* Learning rate: 0.001
* Epochs: 25
* Keep probability: 0.7

I tried increasing the epoch number (50 and 100) but my model seemed to top out around 12-15ish epochs before stabilizing around epoch 20.

I tried a lower "keep probability" of 0.5 but found that 0.7 resulting in a higher prediction rate. 

I tried a few different learning rates (0.01, 0.002) but none of them performed as well.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 95.7% 
* test set accuracy of 86.6%

For my model, I started with a LeNet architecture as a base and then added dropouts between the fully connected layers. I choose the LeNet architecture for this model since it works for digit classification which is a similar problem to traffic sign classification. They both have a small number of classes. I quickly saw that the LeNet architecture gave good results. However, by itself, it didn't get the model performance that I needed. By adding a total of dropout layers (which come between the fully connected layers) I was able to get an validation set accuracy of over 95%. The dropout layers helped by preventing overfitting on the data set, which made the model's solution more robust.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images were all fairly clear and had good lighting. I had to crop them because originally they had the sign as a small part of the total image and the model wasn't able to correctly predict them. Some of the images have fairly large watermarks which I though might confuse the model but it didn't seem to make a difference.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing      		| Children crossing   									| 
| No entry     			| No entry							|
| 60 km/h					| 60 km/h											|
| Stop sign	      		| Stop sign			 				|
| Priority road			| Priority road							|


The model was able to correctly guess all 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 86.6%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th-13th cell of the Ipython notebook.

For the first image, the model is positive that this is a **"Children Crossing"** sign (probability of 1.0), and it's correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Children crossing  									| 
| 0.0     				| Dangerous curve to the right							|
| 0.0					| Bicycles crossing							|
| 0.0	      			| Beware of ice/snow		 				|
| 0.0				    | End of all speed and passing limits  							|


For the second image, the model is positive that this is a **"No entry"** sign (probability of 1.0), and it's correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry 									| 
| 0.0     				| Bumpy road							|
| 0.0					| Stop							|
| 0.0	      			| Dangerous curve to the right		 				|
| 0.0				    | Bicycles crossing							|

For the third image, the model is positive that this is a **"60 km/h"** sign (probability of 1.0), and it's correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 60 km/h 									| 
| 0.0     				| 80 km/h							|
| 0.0					| 50 km/h				|
| 0.0	      			| Slippery road		 				|
| 0.0				    | No passing						|


