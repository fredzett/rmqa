import streamlit as st
from ch1_linear_regression import linear_regression

chapters = ["Linear regression", "Logistic regression"]
ch_selected = st.sidebar.selectbox("",chapters)

# Sidebar
if ch_selected == "Linear regression":
    linear_regression()

