'''
Script for cropping and rotating fish based on their location in the image, found through an object detector
'''
from __future__ import print_function
from skimage.data import imread
from skimage.io import imshow,imsave
from skimage import img_as_float
import pandas as pd
import numpy as np
import cv2
from skimage.util import crop
from skimage.transform import rotate
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import json
import os


def deg_angle_between(x1,y1,x2,y2):
    from math import atan2, degrees, pi
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return degs


def get_rotated_cropped_fish(img, x1, y1, x2, y2):
    (h, w) = img.shape[:2]
    # calculate center and angle
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    angle = np.floor(-deg_angle_between(x1, y1, x2, y2))
    # print('angle=' +str(angle) + ' ')
    # print('center=' +str(center))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    fish_length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    cropped = rotated[(max((center[1] - fish_length / 1.8), 0)):(max((center[1] + fish_length / 1.8), 0)),
              (max((center[0] - fish_length / 1.8), 0)):(max((center[0] + fish_length / 1.8), 0))]
    # imshow(img)
    # imshow(rotated)
    # imshow(cropped)
    resized = resize(cropped, (224, 224))
    return resized


if __name__ == '__main__':
    label_files = ['/home/leo/Fish/train/BET/bet_labels.json',
                   '/home/leo/Fish/train/ALB/alb_labels.json',
                   '/home/leo/Fish/train/YFT/yft_labels.json',
                   '/home/leo/Fish/train/DOL/dol_labels.json',
                   '/home/leo/Fish/train/SHARK/shark_labels.json',
                   '/home/leo/Fish/train/LAG/lag_labels.json',
                   '/home/leo/Fish/train/OTHER/other_labels.json']

    data_dirs = ['/home/leo/Fish/train/BET/',
                 '/home/leo/Fish/train/ALB/',
                 '/home/leo/Fish/train/YFT/',
                 '/home/leo/Fish/train/DOL/',
                 '/home/leo/Fish/train/SHARK/',
                 '/home/leo/Fish/train/LAG/',
                 '/home/leo/Fish/train/OTHER/']
    images = list()
    labels_list = list()
    for c in range(7):
        print(label_files[c])
        labels = pd.read_json(label_files[c])
        for i in range(len(labels)):
            try:
                img_filename = labels.iloc[i,2]
                print(img_filename)
                l1 = pd.DataFrame((labels[labels.filename==img_filename].annotations).iloc[0])
                print('success1')
                image = imread(data_dirs[c]+img_filename)
                print('success2')
                images.append(get_rotated_cropped_fish(image,np.floor(l1.iloc[0,1]),np.floor(l1.iloc[0,2]),np.floor(l1.iloc[1,1]),np.floor(l1.iloc[1,2])))
                print('success3')
                labels_list.append(c)
            except:
                pass

    for i in range(len(images)):
        imsave('../preprocessed_train/img_'+str(i)+'label_'+str(labels_list[i])+'.jpg',images[i])