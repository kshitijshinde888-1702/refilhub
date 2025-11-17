import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Customer Interest Classification + Confusion Matrix")

df = pd.read_csv("data/ReFillHub_SyntheticSurvey.csv")

df['Interest'] = (df['Willingness_to_Pay_AED'] > df['Willingness_to_Pay_AED'].median()).astype(int)

X = df.drop(['Interest'], axis=1).select_dtypes(include=['int64','float64'])
y = df['Interest']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))
