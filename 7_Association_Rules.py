import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Association Rule Mining")

df = pd.read_csv("data/ReFillHub_SyntheticSurvey.csv")

binary_df = df.apply(lambda x: (x > x.mean()).astype(int))

frequent_items = apriori(binary_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)

st.write("### Association Rules")
st.dataframe(rules.sort_values('lift', ascending=False).head(10))
