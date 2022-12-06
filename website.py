import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import streamlit as st
from PIL import Image


def take_picture():
    pic = st.camera_input("Take a picture of your hand")
    pic = cv2.resize(pic, (150, 150))
    pic = pic / 255.0

labels = ['paper', 'scissors', 'rock']

acc = Image.open("correctness.png")
loss = Image.open("loss.png")
report = pd.read_csv('report.csv', index_col=0)

evaluation = pd.read_csv('eval.csv')
evaluation = evaluation.columns.tolist()
test_acc = round(float(evaluation[1]), 4) * 100
test_lost = round(float(evaluation[0]), 4) * 100

col1, col2 = st.columns((1, 1), gap="medium")

with col1:
    st.image(acc, clamp=True)
    st.metric('Test Accuracy', f'{test_acc}%')
with col2:
    st.image(loss, clamp=True)
    st.metric('Test loss', f'{test_lost}%')

st.table(data=report)

img_file_buffer = st.camera_input("Take a picture of your hand")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.image(img)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    st.image(img)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(img.shape)

    # interpreter = tf.tflite.Interpreter(model_path='model.tflite') #allocate the tensors
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    #
    # # Preprocess the image to required size and cast
    # input_shape = input_details[0]['shape']
    # input_tensor = np.array(np.expand_dims(img, 0))
    # input_index = interpreter.get_input_details()[0]["index"]
    # interpreter.set_tensor(input_index, input_tensor)
    # interpreter.invoke()
    # output_details = interpreter.get_output_details()
    #
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # pred = np.squeeze(output_data)
    # highest_pred_loc = np.argmax(pred)
    # label_name = labels[highest_pred_loc]
    #
    # st.write(label_name)