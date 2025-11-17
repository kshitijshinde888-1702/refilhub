import streamlit as st
import pandas as pd

st.title("Dataset Overview")

df = pd.read_csv("data/ReFillHub_SyntheticSurvey.csv")

st.write("### First 10 Rows")
st.dataframe(df.head(10))

st.write("### Dataset Summary")
st.write(df.describe())
