# JavaMNISTCNN

This repository contains a basic convolutional neural network made in order to recognize MNIST digits, implemented from scratch using Java. It also has a framework (also developed from scratch using Java) for evaluating active learning strategies applied on the CNN. 

The work on the active learning framework is available at https://safdarfaisal.github.io/JavaMNISTCNN/.

### Model used for the CNN

MxM input image (28x28 for MNIST digits)
NxN Conv Layer with K filters
PxP Maxpool Layer
Softmax activation

### Results

~85% average accuracy observed on training and test datasets with 30,000 iterations, 3x3 Conv Layer with 12 filters and 2x2 maxpooling, on dataset downloaded from https://www.kaggle.com/datasets/jidhumohan/mnist-png.

### Credits

1. Java matrix library [EJML](http://ejml.org/wiki/index.php?title=Main_Page) 
2. [Apache Commons Lang Library](https://safdarfaisal.github.io/JavaMNISTCNN/)
