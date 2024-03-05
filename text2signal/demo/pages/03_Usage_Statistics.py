import streamlit as st
import collections
import pandas as pd
#from collections import deque 

st.set_page_config(page_title="Signavio LLM-SandBox Usage Statistics", layout="centered")

image_width=int("400")
image_file_name="images/sap-signavio-logo-colored.svg"
with st.sidebar:
    st.image(image_file_name, width = image_width) 

st.sidebar.markdown(f"## Signavio LLM SandBox connections Statistics")
st.markdown(f"### Signavio LLM SandBox connections Statistics")

def password_entered_statistics():
  def login_form():
    st.text_input("Password", type="password", key="password_statistics")
    if st.session_state["password_statistics"] == "u128str":
      st.session_state["statistics_password_correct"] = True
    else:
      st.session_state["statistics_password_correct"] = False
      
  if st.session_state.get("statistics_password_correct", False):
        return True
      
      # Show inputs for username + password.
  login_form()
  if "statistics_password_correct" in st.session_state:
        st.error("😕 Password incorrect")
  return False

if not password_entered_statistics():
    st.stop()

def show_row():
      number= st.session_state.row_number
      ls_llm_row=list(collections.deque(st.session_state.llm_usage_list))[number]["llm_usage"]
      st.session_state.show_row=eval(ls_llm_row)
      
#if st.button("Get Sessions"):
  
if "session_id_list"  in st.session_state:
    ls = list(collections.deque(st.session_state.session_id_list))
    # st.write(f"\"{ls}\" popped from the queue.")
    st.markdown(f"## All Sessions {len(ls)}")
    st.write(pd.DataFrame(ls))
    
if "success_session_id_list"  in st.session_state:
    ls_success=list(collections.deque(st.session_state.success_session_id_list))
    st.markdown(f"## Successful Sessions {len(ls_success)}")
    st.write(pd.DataFrame(ls_success))
    
    
if "llm_usage_list" in st.session_state:
    ls_llm_sessions=list(collections.deque(st.session_state.llm_usage_list))
    st.markdown(f"## LLM Prompts {len(ls_llm_sessions)}")
    df = pd.DataFrame(ls_llm_sessions)
    #st.write(pd.DataFrame(ls_llm_sessions))
    st.data_editor(df)

#if st.button("Get details"):    
    number = st.number_input('Select row for details', key="row_number", min_value=0, max_value=len(ls_llm_sessions)-1, value=0, step=1, 
                             on_change=show_row)

if "show_row" in st.session_state:
      df = pd.DataFrame(st.session_state.show_row)
      st.table(df)
      
       

    