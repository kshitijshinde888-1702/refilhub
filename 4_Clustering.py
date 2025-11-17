import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Customer Segmentation (Clustering)")

df = pd.read_csv("data/ReFillHub_SyntheticSurvey.csv")

numeric = df.select_dtypes(include=['int64','float64'])

scaler = StandardScaler()
scaled = scaler.fit_transform(numeric)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

st.write("### Clustered Data")
st.dataframe(df.head())

fig, ax = plt.subplots()
ax.scatter(scaled[:,0], scaled[:,1], c=df['Cluster'])
ax.set_title("Customer Clusters")
st.pyplot(fig)
