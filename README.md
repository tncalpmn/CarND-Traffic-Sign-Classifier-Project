# **Project 2 - Traffic Sign Recognition**

## Project Definition

#### This project is from the second part of Udacity's Self Driving Car Nanodegree Program and the goal is to recognise the German Traffic Signs by using deep learning algorithms with Tensorflow.

---

### Project Folder

Short overview about which file contains what:
* signnames.csv : Contains the meaning of sign labels
* Traffic_Sign_Classifier.ipynb : contains my framework to train and test the data as Jupyter Notebook
* NewImages : Random 8 images from internet in order to check the accuracy of new data on the trained model
* Traffic_Sign_Classifier.html : Html version of my framework run on Jupyter Notebook
* ImagesForWriteUp : contains images that are attached to this markup file

---
### Aftermath - What I have used?

Here is a list of API functions and python features that I have been using along this project (as a future reference tom myself).
* **pickle** -> is used to bundle  and save the variables(such as Dictionaries, DataFrames) for later use. Data retrieved from one session can be easily called (pickle.load(data)) in current run.
* **pandas** -> Data analysis tool for handling and manipulation data.
* **Series[<condition with Series>]** : filter given Series according to the condition
* **Series.value_counts()** -> returns the unique classes of given array
* **DataFrame.hist()** -> show the histogram of a given DataFrame
* **WordCloud** -> visualisation tool to show fr
* **'-'.join(str(e) for e in list1)** -> join every element in list with given string and return as one
* **shuffle(x,y)** -> shuffles given datasets by keeping the same indexing
* **DataFrame.sort_values()** -> returns sorted dataFrame by a given feature
* **Axes.barh()** -> Draws a vertical bar chart
* **PIL** -> Image manipulation Library
* **np.stack()** -> joins a sequence of arrays along a new axis
* **tf.Variable()** -> to define variable in Tensorflow
* **tf.truncated_normal()** -> assigns random values from truncated normal distribution.
* **tf.nn.conv2d()** -> Applies CNN to given Image datasets
* **tf.nn.relu()** -> An activation function as a perceptron
* **tf.nn.max_pool()** -> Applies Max Pooling to reduce the dimension of Input
* **flatten()** -> Given multiple dimension Array (x,y,z) returns flatted version as (x*y*z,1)
* **tf.matmul()** -> Matrix Multiplicaiton of Tensorflow
* **tf.placeholder()** -> Placeholders are the ones that are fed by real data
* **tf.one_hot()** -> returns values in matrix form in order to avoid highest number significance during training
* **tf.nn.softmax_cross_entropy_with_logits()** -> given real and estimated labels computes cross entropy (will be removed in future)
* **tf.reduce_mean()** -> computes the mean of given Tensor
* **tf.nn.l2_loss()** -> computes L2 regularisation
* **tf.train.AdamOptimizer()** -> An optimizer that adjust learning rate adaptively
* **tf.argmax()** -> return the index of largest value across axis
* **tf.cast()** -> datatype casting in Tensorflow
* **tf.get_default_session()** -> returns current Session if there is one opened already
* **"String {:.3f}".format(var)** -> replaces {} by given variable
* **"tf.train.Saver().save(sess, '')** -> saves model from current session to given path
* **"tf.train.Saver().restore(sess, tf.train.latest_checkpoint('.'))** -> loads model to session from given path
* **tf.nn.softmax()** -> returns softMax Probabilities given logits
* **tf.nn.top_k()** -> returns indices of top k elements from Tensor

---
[//]: # (Image References)

[image1]: ./ImagesForWriteUp/1.png "Visualization"
[image2]: ./ImagesForWriteUp/2.png "Grayscaling"
[image3]: ./ImagesForWriteUp/3.png "Random Noise"
[image4]: ./ImagesForWriteUp/4.png "Traffic Sign 1"
[image5]: ./ImagesForWriteUp/5.png "Traffic Sign 2"
[image6]: ./ImagesForWriteUp/6.png "Traffic Sign 3"
[image7]: ./ImagesForWriteUp/7.png "Traffic Sign 4"
[image8]: ./ImagesForWriteUp/8.png "Traffic Sign 5"
[image9]: ./ImagesForWriteUp/9.png "Wrong Estimate"
[image10]: ./ImagesForWriteUp/10.png "New Distr"
[image12]: ./ImagesForWriteUp/12.png "RotatTransl"


### Framework

1. Import Training, Validation and Test Data from pickle
2. Preprocess data by converting to grayscale and normalising (zero mean)
3. Generate augmented data and merge them with original data for less frequent classes (excluded)
4. Apply LeNet Image Classification Algorithm and generate logits (estimated outputs)
5. Calculate cross entropy by applying Softmax and Cross Entropy to real labels and logits
6. Calculate cross entropy mean
7. Calculate L2 regularisation on output layer weight -> penalize high value weights
8. Calculate the mean again
9. Apply the optimiser to reduce the loss with given epochs and batch size
10. Calculate precision and evaluate first on evaluation dataset then training and finally one time on test data
11. Try trained model on images from external sources and analyse performance
12. Visualise new signs and their softmax values -> with what percentage is the model guess their labels


### Data Set Summary & Exploration

#### 1. Here is some Information about Dataset

Dataset contains examples from 43 distinct German traffic signs in 32x32x3 format and download link can be found [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Labels are also to find in file signnames.csv

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32x3
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.


Here are some examples from dataset...

![alt text][image1]

Distribution of each label among training dataset...

![alt text][image2]

WordCloud Visualisation of frequent used words among label dataset, just for fun...

![alt text][image3]

Why not a bar chart of distributions of most 10 frequent signs among different datasets...

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale otherwise training process would take much longer since there will be 3 channels to take care of. Therefore I have simplified the data by doing so.

Secondly, I have normalised the data, hence I did not want my classifier to focus on the intensity degrees of the pixels. I wanted my images to be handled the same, meaning, a darker and lighter image of Stop Sign should be classified equally at the classifier level.

Here is an example of a traffic sign after normalising and grayscaling:

![alt text][image5]

As suggested, I decided to generate additional data in order to increase the accuracy of the model. In training set there are signs that has less sample than the others. That could result that this signs could not recognise as well as the others. Therefore, I decided to generate fake data that has less than 750 samples in training set by translating and rotating:
~~~~
if label has less that 750 sample:
  translate image by 3 pixels in x direction
  if index odd:
    rotate by -5
  else:
    rotate by 5
  normalise
  add to dataset with corresponding label
~~~~

Here are four examples of randomly rotated and translated augmented images:

![alt text][image12]

The distribution after adding the augmented images to training set is as following...

![alt text][image10]

However, I did not see much improvement at the accuracy after training therefore I decided to exclude them in order to have a faster training process with following switch.
~~~~
if False: # Switch To Use Generated Fake Data
    N_X_train = N_X_train_extended
    y_train = y_train_extended
~~~~

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, Valid, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, Valid, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten      	|outputs 400				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout					|	drop	0.5 during training 										|
|  Fully connected		| outputs 84  								|
| RELU					|												|
|  Fully connected		| outputs 43								|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer by feeding it with L2 regularized weight of last layer:

regularizer = tf.reduce_mean (loss_cross_entropy + beta * tf.nn.l2_loss(last_weight))

What beta here does is to penalise the high value weights. To achieve the best validation accuracy, I selected my hyper-parameters accordingly:
~~~~
beta = 0.05 (for L2 Regularization)
learn_rate = 0.0002 (for AdamOptimizer)
EPOCH = 25
BATCH_SIZE = 100
keep_prob = 0.5 (for Dropout)
~~~~
Here is validation accuracy on 25 trained model:

![alt text][image6]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

After tuning my hyper-parameters I have achieved following final results:

* training set accuracy of 99.7
* validation set accuracy of 95.5
* test set accuracy of 93.4

High accuracy on training set first might have interpreted as overfitting, but since the validation/test accuracy was also satisfying, I was convinced with the performance of the model.

I have selected the well known LeNet Architecture on this project with some minor adjustments like Dropout and L2 regularisation. This architecture is selected due to its simplicity and high accuracy on image classification problems.

Just adding dropout has increased validation accuracy significantly (up to %3) and L2 Regularisation had also contributed few percentage on the accuracy. One thing to mention is that adding multiple Dropout Layer had an negative impact on the precision and I assume that this is because too many significant features are casted off and that lead to prevent the model to learn more.

High beta (0.05 rather than 0.01) of L2 regularisation had also positive effect on the accuracy.

One last thing to mention is that instead of increasing EPOCH or learning_rate decreasing BATCH_SIZE lead to a better performance.

The size of images (32x32) was enough to be understood by the model (also by human). There were no necessity for high resolution images of signs in order to determine their labels for better accuracy. That makes LeNet Architecture cut out for this classification task -> Easy to modify & efficient.

Despite LeNets satisfying performance on this classification task, it is a quite old method (1998) and I believe new/better architectures could be implemented to getter accuracies. A summary of novel approaches (also as future reference to myself) could be found [here](http://slazebni.cs.illinois.edu/spring17/lec01_cnn_architectures.pdf)

Next I am planning to implement GoogLeNet or ResNet architecture to another image classification task.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight preprocessed German traffic signs that I found on the web :

![alt text][image7]

I have intentionally selected 8 images that might be hard for classifier to detect.

1. Speed Sign (60km) - Might be difficult to identify because this sign has many other similar examples in the dataset (30km, 50km, 80km .. )
2. Children Crossing - Might be difficult to identify because in training set it has few data.
3. Ahead  only - A sign with arrow could be easily mixed up with another arrow sign.
4. Others - were randomly chosen, only shape of the signs
  - Square -> Priority Road
  - Triange -> Yield
  - Hexagon - Stop
  - Circle - No Entry
  , was taken into consideration.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| No Entry     			| No Entry 										|
| Yield					| Yield											|
| Stop 	      		| Stop 					 				|
| Priority Road			| Priority Road      							|
| Ahead  only			| Ahead  only      							|
| Speed Sign		| Speed Sign*      							|
| Children Crossing			| Children Crossing*     							|


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ~%93.4.

Note: It is observed that the model trained with exact same hyper-parameters (HP) result in to different validation percentage and consequently to worse estimation on the new images(87.5). This is simply because of the randomness behaviour of the Architecture (example: Dropout) and also to the shuffling process in each Epoch. As expected some models with same HPs had difficulties guessing Children Crossing and Speed Sign. Here is a screenshots of how one of the models guessed the Children Crossing sign wrong.

![alt text][image9]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

Here is the first 5 image predictions with corresponding softmax probabilities with the last model I trained (note that with the last model accuracy on the new test images was %100):

![alt text][image8]
