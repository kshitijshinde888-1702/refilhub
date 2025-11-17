import streamlit as st
from PIL import Image

st.set_page_config(page_title="ReFill Hub – Smart Refill Stations", layout="wide")

logo = Image.open("assets/refillhub_logo.png")
st.image(logo, width=220)

st.title("ReFill Hub – Smart Refill Stations Dashboard")
st.write("Welcome to the analytics platform. Use the sidebar to navigate through insights and models.")
