# C3D
Pytorch implementation of [Learning Spatialtemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767.pdf) with temporal depth *d=3*. 

## The model 
C3D is a Convolution Neural Network with a 3 dimensional (3 x 3 x 3) kernel which learns spatial-temporal information across multiple frames in the same video clip. 

## Requirement 
1. Pytorch 1.0 or above 
2. OpenCV 4.1 or above

## Usage

## Result

## Notes

### Learning rate decay 
A small step size for the learning rate decay might result in a lower accuracy as the learning rate might have decayed too early. And as a result, the loss might step into a local minima. 

### Data augmentation 
The training data is randomly flip horizontally with a probability of 0.5, resulting in data augmentation which contributes to a significant increase in terms of the accuracy of the model trained. 