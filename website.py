import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
acc = Image.open("correctness.png")
loss = Image.open("loss.png")
report = pd.read_csv('report.csv')

st.image(acc, clamp=True)
st.image(loss, clamp=True)
st.dataframe(data=report)






