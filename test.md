## Active Learning and using it on a CNN

What is actve learning? How does it help? Where can we use it?

Active learning refers to a group of methods in machine learning that allow a model to learn using unlabelled data that is provided during the training process. The model learns how to prioritize labelling data points based on existing labelled data. There are multiple ways in which this labelled data can be queried by the model. They include:
1.	Random Sampling: A random subset of the data is chosen for labelling
2.	Stratified Sampling: Data is divided into groups based on common characteristics. Subsets from these groups are taken
3.	Cluster Sampling: Data is divided into clusters. Subsets of clusters are taken for training
4.	Uncertainty Sampling: Data points are picked based on the least confidence metric
5.	Diversity Sampling: Data points are picked in a manner that maximises entropy
6.	Representative Sampling: Samples taken represent the overall distribution of the data
7.	Redundancy Sampling: Data points more distinct from pre-existing points will be preferred.
8.	Query by committee: Samples selected are those that provide different outputs for multiple models.

Active Learning helps to reduce the effort needed to generate labels for every datapoint in the dataset since the model itself begins to label the datapoints based on a small pool of the original dataset.

My aim when I started this project was to answer the first three questions. This was to be done by implementing different methods for active learning and testing to see if the accuracy was comparable to that seen with all the points already labelled.

### The methods explored here 

There are many methods when it comes to active learning algorithms. A few of which are expanded upon in this section.

**Least Confidence Method:** Here the samples that need to be labelled are selected when the model gives the lowest confidence probability for the output class. This is done over multiple Iterations/stages to generate labelled data. 

$$\hat{x} = \underset{i}{min}\left(\underset{\theta}{argmax}\left(P(y | x \right)\right)$$

One advantage of the method of least confidence is that it is simple and easy to implement. It only requires the model to make predictions on the unlabeled samples and select the one with the lowest predicted probability for the correct class. However, it also has some limitations. For example, it may not always select the most informative samples, particularly if the model has high uncertainty or is not well-calibrated. It may also be susceptible to noise or outliers in the data. Overall, the method of least confidence can be a useful technique for uncertainty sampling in some situations, but it may not be the most effective approach in all cases.


**Smallest Margin Sample:** Smallest margin sampling is a technique for uncertainty sampling in active learning where the goal is to select samples where the model has the smallest margin of confidence in its predictions. This approach assumes that samples where the model is less confident are more likely to be informative and improve the model's performance.

$$x = \underset{x}{\argmin}\,P_\theta(y_1|x) - P_\theta(y_2|x)$$

Here, $y_1$ refers to the class with the highest confidence and $y_2$ refers to the class with the second highest confidence.

To implement smallest margin sampling, the model is first applied to the pool of unlabeled samples and the predicted probabilities for each sample are calculated. The sample with the smallest difference between the predicted probability for the correct class and the predicted probability for the next-highest class is then selected for labeling. This process is repeated until a sufficient number of samples have been labeled and the model can be updated.

**Largest Margin Sampling:** Largest margin sampling is a technique for uncertainty sampling in active learning where the goal is to select samples where the model has the largest margin of confidence in its predictions. This approach assumes that samples where the model is more confident are more likely to be informative and improve the model's performance.

To implement largest margin sampling, the model is first applied to the pool of unlabeled samples and the predicted probabilities for each sample are calculated. The sample with the largest difference between the predicted probability of the class with highest confidence and the predicted probability for the class with the lowest confidence is then selected for labeling. This process is repeated until a sufficient number of samples have been labeled and the model can be updated.

**Entropy:** Entropy sampling is a technique for uncertainty sampling in active learning where the goal is to select samples where the model has the highest uncertainty or randomness in its predictions. This approach assumes that samples where the model is more uncertain are more likely to be informative and help reduce the uncertainty.

$$\mathrm{Entropy} = \underset{i}{\sum} \left(-P_\theta(y|x)\cdot\log P_\theta(y | x)\right)$$

To implement entropy sampling, the model is first applied to the pool of unlabeled samples and the predicted probabilities for each sample are calculated. The sample with the highest entropy, which is a measure of the uncertainty or randomness in the model's predictions, is then selected for labeling. This process is repeated until a sufficient number of samples have been labeled and the model can be updated.

**Query by committee:** Query by committee is a technique for selecting samples for active learning where multiple models are trained on the same data and their predictions are compared. The idea is that if multiple models make different predictions on a particular sample, it is likely to be more informative and useful for training the overall model.

To implement query by committee, the data is first divided into a training set and a pool of unlabeled samples. Multiple models are then trained on the training set. When it is time to select a new sample for labeling, the models are applied to the pool of unlabeled samples and their predictions are compared. If there is disagreement among the models (i.e. they make different predictions on the same sample), that sample is selected for labeling. This process is repeated until a sufficient number of samples have been labeled and the overall model can be updated.

**Algorithms used in QBC:**  
a) **Vote Entropy**: Vote entropy is a measure of the uncertainty or randomness in the models' predictions. It is calculated by taking the sum of the negative of the predicted probabilities for each class, multiplied by the log of the predicted probabilities. A high vote entropy indicates that the models have high uncertainty in their predictions, whereas a low vote entropy indicates that the models are more confident and consistent in their predictions.

b) **KL Divergence:** KL divergence, also known as Kullback-Leibler divergence, is a measure of the difference between two probability distributions. In the context of QBC, it is used to compare the predicted probabilities of the models for each sample. A high KL divergence indicates that the models have significantly different predicted probabilities, whereas a low KL divergence indicates that the models have similar predicted probabilities.


### The Dataset
The dataset that we have decided to use is the MNIST dataset. The MNIST dataset is a set of handwritten digits of size 28x28. The dataset was taken from Kaggle. It consists of 60,000 images for the training dataset and 10,000 for the test dataset. Each image is labelled with a value between zero to nine. 

>![2](dataset image)
>
>**A set of sample images from MNIST (From Wikipedia)**

Models designed to tackle this classification problem are normally those used in image processing systems. They attempt to decipher handwritten images and convert it into digits	. An extended version of MNIST known as EMNIST also provides images of uppercase and lowercase alphabets which is kept in the same 28x28 pixel format. I however keep my analysis to just the original MNIST dataset.

### The benchmark model

[1][(image.png)]

We plan to use [a basic CNN model](https://github.com/safdarfaisal/CNN_new) with the following components.
1.	8 convolution filters with a 3x3 kernel and a stride length of 1
2.	A 2x2 max pool layer 
3.	A single layer fully connected network
4.	Softmax activations
Upon training the model with 10000 images using just passive learning, we were able to generate an accuracy of around 85%.

### Results

Our major focus looked at the heuristic methods of using active learning.

In this we noted that some methods fared considerably better than others and were all comparable to the passively learnt model with much lower levels of effort.

![Results][(test.jpg)]

As you can see from this graph, a few methods showed an abnormal behavior when accuracy was used as a metric for measurement. The major one showing this phenomenon were QBC with max vote entropy as the metric. We also noticed similar issues with KL divergence where only a single class was takem to be used for training.

### Discussion

We have seen that active learning methods are very powerful tools that can be used to augment our models providing more value for the similar or lower effort. There are a few issues that need further study - so as to why methods like QBC or KL divergence did not semm to work as intended. This may have been due to the lack of variance between models that are trained on similar data or could be due to a lack of variance being observed. 
