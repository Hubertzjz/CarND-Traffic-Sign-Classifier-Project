# **Traffic Sign Recognition** 

## Project Writeup - J. Zhao

### 2019/10/23

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/chart.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./GTSRB_test_images/00000.jpg "Traffic Sign 1"
[image4]: ./GTSRB_test_images/00001.jpg "Traffic Sign 2"
[image5]: ./GTSRB_test_images/00002.jpg "Traffic Sign 3"
[image6]: ./GTSRB_test_images/00003.jpg "Traffic Sign 4"
[image7]: ./GTSRB_test_images/00004.jpg "Traffic Sign 5"
[image8]: ./GTSRB_test_images/00005.jpg "Traffic Sign 6"
[image9]: ./GTSRB_test_images/00006.jpg "Traffic Sign 7"
[image10]: ./GTSRB_test_images/00007.jpg "Traffic Sign 8"
[image11]: ./GTSRB_test_images/00008.jpg "Traffic Sign 9"
[image12]: ./GTSRB_test_images/00009.jpg "Traffic Sign 10"


---

### Data Set Summary & Exploration

#### 1. The basic summary of the data set.

I used the python & numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many samples in the training data set per label.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre-processing techniques.

As a first step, I decided to convert the images to grayscale as applied in LeNet's Lab, which appears to have better starting accurancy than RGB images. I've also tried turning images to HSV but it's difficult to train.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then, I shifted and normalized the image data because zero-mean and equal-variance will yield better training performance.

#### 2. Model architecture.

My final model consisted of the following layers (LeNet-5 with dropout layers):

| Layer         		|     Description	        			| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray-scale image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flaten	      	| outputs 400 				              |
| Dropout          |                                |
| Fully connected		| outoputs 120 									|
| RELU					|												|
| Dropout          |                                  |
| Fully connected		| outoputs 84 									|
| RELU					|												|
| Dropout          |                                 |
| Fully connected		| outoputs 43 									|


#### 3. Optimizer & training parameters.

To train the model, I used the AdamOptimizer as applied in LeNet's Lab. The L2 regularization is added to deal with overfitting.
Hyperparameters are as follows:

| Hyperparameter      		|     Value	        					| 
|:---------------------------:|:---------------------------------------------:| 
| Number of epochs        | 30   							          | 
| Batch size     	     | 200                            	|
| Learning rate		    	 | 0.001									   |
| μ, σ	      	       | 0, 0.1 			                 	|
| Keep probability	        | 0.8	                          |
| L2 Regularization coefficient	| 0.01										|

#### 4. Final solution.

My final model results were:
* validation set accuracy of 0.955 
* test set accuracy of 0.942

The first approach is to play round with number of epochs, batch size and learning rate as in the LeNet's Lab. The best validation accurancy was about 0.92. In order to imporove the accurancy, a L2 regularization term is imported into the loss. The final step is to deal with overfitting with an early stop of loss, several dropout layers are added among the fully connected layers. It turns out that the test set accurancy is similar with validation set accurancy, which means that the model works fine with real data.  


### Test a Model on New Images

#### 1. German traffic signs.

Here are ten German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12]

The second image maybe difficult to recognize because the traffic sign is out of shape.

#### 2. Predictions on new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited      		| Vehicles over 3.5 metric tons prohibited   									| 
| Speed limit (30km/h)     			| Turn right ahead 										|
| Keep right				| Keep right											|
| Turn right ahead	      		| Turn right ahead					 				|
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|
| Keep right			| Keep right      							|
| Priority road		| Priority road      							|
| Road work			| Road work     							|
| Ahead only		| Ahead only     							|
| Priority road			| Priority road     							|


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 94.2%.

#### 3. Softmax probabilities for each prediction.

The code for generating probabilities from predictions on my final model is realized with `tf.nn.top_k` function.

For the first image, the model is recognized that this is a vehicles over 3.5 metric tons prohibited sign (probability of 0.975), and it truely exist this traffic sign in the image. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9757    			| Vehicles over 3.5 metric tons prohibited   				| 
| .0086    				| Speed limit (100km/h) 								|
| .0071					| No passing											|
| .0032					| Speed limit (30km/h)									|
| .0012					| Ahead only											|

For the second image, the model is convinced that this is a turn right ahead sign (probability of 0.8940), the image acutally is a speed limit (30km/h) sign which is predicted with probablity of 0.0086.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .8940         			| Turn right ahead   									| 
| .0968     				| Stop 										|
| .0086					| Speed limit (30km/h)											|
| .0005					| Roundabout mandatory											|
| .00004					| Speed limit (80km/h)											|

For the other 8 images, similar with 1st image, all the traffic signes are corectly recognized with probablity higher than 0.9516.
