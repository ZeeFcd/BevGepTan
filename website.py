import numpy as np
from keras.utils import load_img, img_to_array
import streamlit as st

acc = load_img("correctness.png")
acc = img_to_array(acc)

loss = load_img("loss.png")
loss = img_to_array(loss)

st.image(acc)
st.image(loss)







