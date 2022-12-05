import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


acc = Image.open("correctness.png")
loss = Image.open("loss.png")
report = pd.read_csv('report.csv', index_col=0)
evaluation = pd.read_csv('eval.csv')
test_loss = evaluation[0]
test_acc = evaluation[1]

col1, col2 = st.columns((1, 1))

with col1:
    st.image(acc, clamp=True)
with col2:
    st.image(loss, clamp=True)

col4, col5 = st.columns((1, 1))

with col4:
    st.metric('Test Accuracy', test_acc)
with col5:
    st.metric('Test loss', test_loss)

st.table(data=report)


