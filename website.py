import numpy as np
import cv2
import pandas as pd
import tensorflow.lite as tflite
import streamlit as st
from PIL import Image
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.float32(img)

    interpreter = tflite.Interpreter(model_path='model.tflite') #allocate the tensors
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image to required size and cast
    input_shape = input_details[0]['shape']
    input_tensor = np.array(np.expand_dims(img, 0))
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    highest_pred_loc = np.argmax(pred)
    label_name = labels[highest_pred_loc]

    st.write(label_name)