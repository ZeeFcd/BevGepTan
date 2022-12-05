import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


acc = Image.open("correctness.png")
loss = Image.open("loss.png")
report = pd.read_csv('report.csv', index_col=0)

evaluation = pd.read_csv('eval.csv')
evaluation = evaluation.columns.tolist()
test_acc = round(float(evaluation[1]), 4) * 100
test_lost = round(float(evaluation[0]), 4) * 100

col1, col2 = st.columns((1, 1))

with col1:
    col4, col5 = st.columns((1, 1))
    with col4:
        st.image(acc, clamp=True)

    with col5:
        st.image(loss, clamp=True)
    st.table(data=report)

with col2:
    st.metric('Test Accuracy', f'{test_acc}%')
    st.metric('Test loss', f'{test_lost}%')



