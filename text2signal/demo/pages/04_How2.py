import streamlit as st
import collections
import pandas as pd
#from collections import deque 

st.set_page_config(page_title="Signavio LLM-SandBox Helper", layout="centered")

image_width=int("400")
image_file_name="images/sap-signavio-logo-colored.svg"
with st.sidebar:
    st.image(image_file_name, width = image_width) 

st.sidebar.markdown(f"## Signavio LLM-SandBox Helper")
st.markdown(f"### Signavio LLM-SandBox Helper")


st.markdown(f"#### Where to find workspace ID? - Processes -> demo01 -> Process Settings -> Tenant ID")
image_file_name="images/help_00.png"
st.image(image_file_name, width = image_width) 