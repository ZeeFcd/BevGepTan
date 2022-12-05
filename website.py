import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


acc = Image.open("correctness.png")
loss = Image.open("loss.png")
report = pd.read_csv('report.csv', index_col=0)
evaluation = pd.read_csv('eval.csv')

col1, col2 = st.columns((1, 1))

with col1:
    st.image(acc, clamp=True)
with col2:
    st.image(loss, clamp=True)

col4, col5 = st.columns((1, 1))

#with col4:
#    st.metric('Test Accuracy', value, delta=None, delta_color="normal", help=None)
#with col5:
#   st.metric('Test loss', value, delta=None, delta_color="normal", help=None)

st.table(data=report)


