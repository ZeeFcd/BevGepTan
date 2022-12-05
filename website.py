import numpy as np
from keras.utils import load_img, img_to_array
import streamlit as st

acc = load_img("correctness.png")
acc = img_to_array(acc)
acc = acc / 255.0

loss = load_img("loss.png")
loss = img_to_array(loss)
loss = loss / 255.0

st.image(acc)
st.image(loss)







