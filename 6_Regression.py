import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("Regression – Predict Willingness to Pay (AED)")

df = pd.read_csv("data/ReFillHub_SyntheticSurvey.csv")

target = "Willingness_to_Pay_AED"
X = df.drop(columns=[target]).select_dtypes(include=['float64','int64'])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("### Model Performance")
st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
