import numpy as np
import streamlit as st
from PIL import Image
acc = Image.open("correctness.png")
loss = Image.open("loss.png")

st.image(acc, clamp=True)
st.image(loss, clamp=True)







