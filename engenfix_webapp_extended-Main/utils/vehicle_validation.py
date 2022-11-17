import os
import pickle
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img



vehicle_validation_model = tf.keras.models.load_model('./model/classifier/vehicle validator_model_inceptionv4.h5')



def valid_vehicle(file):


    test = load_img(file, target_size = (299,299))
    test_img = image.img_to_array(test)
    test_img = test_img / 255
    test_img = np.expand_dims(test_img, axis=0)
    # result = vehicle_validation_model.predict(test_img)

    is_valid_pred = np.argmax(vehicle_validation_model.predict(test_img), axis=1)

    if is_valid_pred == 0:
        # print('A Vehicle')
        is_valid_vehicle = True
    else:
        # print('Not a Vehicle')
        is_valid_vehicle = False

    return is_valid_vehicle
