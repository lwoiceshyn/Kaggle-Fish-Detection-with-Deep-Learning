'''
Iterates over all images in the directory

Checks filename

Places into folder corresponding to its filename
'''
import os
import shutil

directory = '/home/leo/Fish/preprocessed_train/'


for filename in os.listdir(directory):
    if filename.endswith("label_0.jpg"):
        shutil.move(directory + str(filename), "/home/leo/Fish/preprocessed_train/BET")
    elif filename.endswith("label_1.jpg"):
        shutil.move(directory + str(filename), "/home/leo/Fish/preprocessed_train/ALB")
    elif filename.endswith("label_2.jpg"):
        shutil.move(directory + str(filename), "/home/leo/Fish/preprocessed_train/YFT")
    elif filename.endswith("label_3.jpg"):
        shutil.move(directory + str(filename), "/home/leo/Fish/preprocessed_train/DOL")
    elif filename.endswith("label_4.jpg"):
        shutil.move(directory + str(filename), "/home/leo/Fish/preprocessed_train/SHARK")
    elif filename.endswith("label_5.jpg"):
        shutil.move(directory + str(filename), "/home/leo/Fish/preprocessed_train/LAG")
    elif filename.endswith("label_6.jpg"):
        shutil.move(directory + str(filename), "/home/leo/Fish/preprocessed_train/OTHER")
    else:
        pass