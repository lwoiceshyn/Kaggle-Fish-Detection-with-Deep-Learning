import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_test_samples = 1000
n_augmentation = 5
img_width = 299
img_height = 299
batch_size = 32

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


root_path = '/home/leo/PycharmProjects/Fisheries/'
weights_path = os.path.join(root_path, 'weights.h5')
test_data_dir = os.path.join(root_path, 'test_stg1/')

# Parameters for Keras ImageDataGenerator
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

#Load Inception V3
print('Loading the pre-trained InceptionV3 model and weights...')
InceptionV3_model = load_model(weights_path)

for i in range(n_augmentation):
    print('{}th Augmentation for Testing...'.format(i))
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = False,
            seed = random_seed,
            classes = None,
            class_mode = None)

    test_image_list = test_generator.filenames

    print('Begin to predict for testing data ...')
    if i == 0:
        predictions = InceptionV3_model.predict_generator(test_generator, n_test_samples)
    else:
        predictions += InceptionV3_model.predict_generator(test_generator, n_test_samples)

predictions /= n_augmentation

#Write final submission file
print('Beginning to write Kaggle submission file')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, n_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated in root path/')
