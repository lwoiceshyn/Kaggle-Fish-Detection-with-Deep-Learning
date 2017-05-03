# Kaggle Fish-Detection with Deep Learning

This repository contains the work I did for The Nature Conservancy Fisheries Monitoring competition on Kaggle. The approach used was to run an object detection algorithm to learn to find the fish in each image, and store these in a JSON file. Then, a script was used to take the fish locations, and modify the datset by cropping and centering the fish.

Transfer learning was done using the Inception-V3 model to learn classification among the fish types.

## Libraries
* Tensorflow 1.X
* Keras
* Numpy

## Examples

Raw Fish Image Example:

![Alt text](Sample%20Images/img_00039.jpg "Raw Fish Image Example")

Cropped Fish Image Example:

![Alt text](Sample%20Images/img_199label_1.jpg "Cropped Fish Image Example")
