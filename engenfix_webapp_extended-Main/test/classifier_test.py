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



# damage_model = tf.keras.models.load_model('model/classifier/model_dmg_inceptionv3.h5', custom_objects={'tf': tf})
# damage_model = tf.keras.models.load_model('model/classifier/vehicle_type_inceptionv4.h5')
damage_model = tf.keras.models.load_model('model/classifier/vehicle_orientation.h5')
# damage_model = tf.keras.models.load_model('model/classifier/model_dmg_inceptionv3.h5', custom_objects={'tf': tf})




test = load_img('/media/shane/New Volume1/office_engenai/engenai/engenfix/code/classification models/Data (2)/Data/test/Back/0563.JPEG', target_size = (299,299))

test_img = image.img_to_array(test)
test_img = test_img / 255
test_img = np.expand_dims(test_img, axis=0)
result = damage_model.predict(test_img)

a = np.argmax(damage_model.predict(test_img), axis=1)

print(a)