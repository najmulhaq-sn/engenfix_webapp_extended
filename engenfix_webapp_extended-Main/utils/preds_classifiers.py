# doing the image pre processing and prediction
# 1. loading the libraries
# 2. loading the models
# 3. image preprocessing - load the img, img to array, resize the image, standardise the image
# 4. do the predictions
# 5. get the preds to a dict
# 6. jsonify and send to the report

# from flask import Flask, flash, request, redirect, url_for, render_template
import os
import pickle
import PIL
from PIL import Image
import numpy as np
# import pandas as pd
# import sklearn
# from glob import glob
# import tensorflow as tf
# from tensorflow import keras
# import tensorflow as tf
# import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
import cv2

# Loading the models

# orientation_model = keras.model.load_model('static/models/inceptionv3mod_hg-vehicle_class.h5')
# type_model = keras.model.load_model('static/models/static/models/inceptionv3mod_hg-vehicle_class.h5')
# damage_model = keras.model.load_model('static/models/model.h5')

# classifier models
type_model = tf.keras.models.load_model('./model/classifier/vehicle_type_inceptionv4.h5')
orientation_model = tf.keras.models.load_model('./model/classifier/vehicle_orientation_model_inceptionv4.h5')
damage_model = tf.keras.models.load_model('./model/classifier/model_dmg_inceptionv3.h5', custom_objects={'tf': tf})




def image_prep(file):
    # result_dict = {'img_type': [], 'img_ori': [], 'img_damage': []}
    result_dict = {}
    # type_model = tf.keras.models.load_model('static/models/vehicle_type.h5')
    # orientation_model = tf.keras.models.load_model('static/models/vehicle_orientation.h5')
    # damage_model = tf.keras.models.load_model('static/models/Incepmodel3.h5', custom_objects={'tf': tf})

    test = load_img(file, target_size = (299,299))
    # test = cv2.imread(file)
    # test = cv2.resize(test,(224,224)) 
    # test = test.reshape(1,224,224,3)
    test_img = image.img_to_array(test)
    test_img = test_img / 255
    test_img = np.expand_dims(test_img, axis=0)
    a = np.argmax(type_model.predict(test_img), axis=1)
    b = np.argmax(orientation_model.predict(test_img), axis=1)
    c = np.argmax(damage_model.predict(test_img), axis=1)

    if a == 0:
        # print('Auto')
        v_type = "Trishaw"
    elif a == 1:
        # print('Bus')
        v_type = "Bus"
    elif a == 2:
        # print('Car')
        v_type = "Car"
    elif a == 3:
        # print('Motorcycle')
        v_type = "Motorcycle"
    elif a == 4:
        # print('Truck')
        v_type = "Truck"
    else:
        # print('Van')
        v_type = "Van"

    if b == 0:
        v_orient = 'Back'
    elif b == 1:
        v_orient = 'Front'
    elif b == 2:
        v_orient = 'Left'
    elif b == 3:
        v_orient = 'Right'
    else:
        v_orient = 'Top'

    if c == 0:
        v_damage = 'low'
    elif c == 1:
        v_damage = 'medium'
    elif c == 2:
        v_damage = 'no_damage'
    else:
        v_damage = 'Severe'

    # result_dict['img_type'].append(v_type)
    # result_dict['img_ori'].append(v_orient)
    # result_dict['img_damage'].append(v_damage)
    result_dict['img_type'] = v_type
    result_dict['img_ori']= v_orient
    result_dict['img_damage']= v_damage


    # print(v_type)
    # print(v_orient)
    # print(v_damage)
    # print(result_dict)

    return result_dict

           # v_damage

# <h4>Vehicle Damage status : {{result_dict['v_damage'][i]}}</h4>
