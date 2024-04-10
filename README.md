# AlexNet Implementation with Tensorflow-Keras

![alt text](https://th.bing.com/th/id/R.b5bd0f70b4bd3c335417112a329fdd56?rik=j3Vwi5BKfTA2WQ&riu=http%3a%2f%2fleiblog.wang%2fstatic%2fimage%2f2020%2f10%2fTL5kcp.jpg&ehk=ukswjG1YAvpxb1nyrcxHvBx4LQw8vplnNQFTk0BQPrA%3d&risl=&pid=ImgRaw&r=0)


## Introduction

This repository contains an implementation of the AlexNet model, a convolutional neural network (CNN) architecture named after Alex Krizhevsky who used it to win the 2012 ImageNet competition. 

## Model Architecture

AlexNet is composed of five convolutional layers, followed by three fully connected layers. It uses ReLU for the nonlinearity functions and employs dropout and data augmentation as regularization methods. As shown in the image , model is seperated into two branches and each branch is trained on a particular GPU , but in our code , we use a single branche sequential model and train the whole model on on GPU.

## Dependencies

This project requires the following libraries:
- TensorFlow
- Keras

## Dataset

The model is trained and tested on the CIFAR10 dataset, which is directly loaded from Keras datasets. Due to limitations, only 1/5 of the whole dataset is used.

## Preprocessing

The images in the dataset are resized to 128x128 pixels to match the input shape of the AlexNet model.

## Training

The model is compiled with the SGD optimizer with a learning rate of 0.001, and the sparse categorical cross-entropy loss function. It is then trained for 64 epochs.

## Results

After 50 epochs of training on 10000 train images and validating 1000 validation images, the model showd 99% accuracy on training set and 66% acuracy on test set ; which is acceptable because we used only 10000 images out of 50000 images for training .

## Usage

1. Clone this repository.
2. Run the Python file AlexNet.oy

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
