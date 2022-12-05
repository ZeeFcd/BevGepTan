import numpy as np
import streamlit
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.utils import load_img, img_to_array

#def predict_hand():
   # img = load_img(os.path.join(folder, image), target_size=(150, 150))
  #  img = img_to_array(img)
   # img = img / 255.0
    #return 0

model = tf.keras.models.load_model('model_saved')




