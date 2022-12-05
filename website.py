import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
acc = Image.open("correctness.png")
loss = Image.open("loss.png")
report = pd.read_csv('report.csv', index_col=0)

col1, col2 = st.beta_columns((1, 1))

with col1:
    st.image(acc, clamp=True)


with col2:
    st.image(loss, clamp=True)

st.table(data=report)


