import streamlit as st
import collections
import pandas as pd
#from collections import deque 

st.set_page_config(page_title="Signavio LLM-SandBox Helper", layout="centered")

image_width=int("400")
image_file_name="notebooks/pages/sap-signavio-logo-colored.svg"
with st.sidebar:
    st.image(image_file_name, width = image_width) 

st.sidebar.markdown(f"## SSignavio LLM-SandBox Helper")
st.markdown(f"### Signavio LLM-SandBox Helper")

