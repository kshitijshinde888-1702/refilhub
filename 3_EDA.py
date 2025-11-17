import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Exploratory Data Analysis (EDA)")

df = pd.read_csv("data/ReFillHub_SyntheticSurvey.csv")

numeric_cols = df.select_dtypes(include=['float64','int64']).columns

st.write("### Numerical Feature Distribution")

for col in numeric_cols:
    fig, ax = plt.subplots()
    ax.hist(df[col], bins=20)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)
