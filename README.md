# intuitive-drug

## Requirements:
* Python 2.7
* Tensorflow 1.2.0
* numpy 1.14.1
* PIL 5.1.0

## Introduction
Apply Convolutional Neural Networks to 2D structures of molecules. 
Models trained before can be found in './models'.

The code is based on the Tensorflow's [tutorial](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist) on mnist dataset.
After decompressing the two compressed files, images for training, testing and screening can be found.

To prepare the dataset:
```
$ python predata.py
```
To train a model:
```
$ python model.py
```
To screen:
```
$ python screen.py
```
All the parameters in these files can be changed.

Though the model is simple, it helped discover two active compounds.
