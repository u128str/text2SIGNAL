# This is first starting page for Text2Signal SandBox UI
# $ streamlit run notebooks/01_gui_signvio_signal_main_page.py 

import streamlit as st
import requests

st.set_page_config(
        page_title="Signavio Connection",
#        page_icon=":-)"
)

image_width=int("400")
image_file_name="notebooks/pages/sap-signavio-logo-colored.svg"
with st.sidebar:
    st.image(image_file_name, width = image_width)    



st.markdown("# Signavio connection tab ")
#image_width=int("400")
#image_file_name="notebooks/pages/sap-signavio-logo-colored.svg"
#st.sidebar.image("/home/rzwitch/Downloads/randy-streamlit.png", use_column_width=True)
#st.sidebar.image(image_file_name, width = image_width)
#st.sidebar.markdown("# Login page")


import hmac
import pandas as pd

#Signavio
from signavio_lib import credentials_actualization, POST_Signavio
from signavio_lib import q_list_processes, create_investigation, list_of_investigations
from signavio_lib import q_list_columns, query_investigation_details, get_process_views, get_workspaces

from datetime import datetime
from collections import deque 
import uuid



if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()


@st.cache_resource
def get_app_queue():
    return deque()

app_queue = get_app_queue()

if "session_id_list" not in st.session_state:
    app_queue.append({"session":f'{st.session_state.session_id}',
        "session_creation_datetime": datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
        }
                      )
    st.session_state.session_id_list=app_queue

st.link_button("Click here to open Signavio original page", "https://editor.signavio.com/g/statics/pi/areas")

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", "alexey.streltsov@sap.com", key="username_signavio")
            st.text_input("Password", type="password", key="password")
            #system_instance = 'https://editor.signavio.com'
            #workspace_id = 'b0f07deabd3140aea5344baa686e0d84' # workspace Process AI 
            #workspace_name="Process AI"
            st.text_input("system_instance", 'https://editor.signavio.com',  key="system_instance")
            st.text_input("workspace_id", "b0f07deabd3140aea5344baa686e0d84", key="workspace_id")
            st.text_input("workspace_name","Process AI",  key="workspace_name")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        #system_instance=st.session_state["system_instance"]
        #workspace_id= st.session_state["workspace_id"]
        #workspace_name= st.session_state["workspace_name"]
        #
        #username= st.session_state["username"]
        #pw=st.session_state["password"]
        try:
            username= st.session_state["username_signavio"]
            workspace_name=st.session_state["workspace_name"]
            auth = credentials_actualization(st.session_state["system_instance"], 
                                                          st.session_state["workspace_id"], 
                                                          username, 
                                                          st.session_state["password"], 
                                                           workspace_name=workspace_name) 
            st.session_state.auth=auth
            st.session_state["workspacename"]=st.session_state["workspace_name"]
            st.session_state["systeminstance"]=st.session_state["system_instance"]
            st.session_state["workspaceid"] = st.session_state["workspace_id"]
            st.session_state.username = username
            try: 
               st.session_state["list_of_workspaces"] = POST_Signavio(query=get_workspaces,workspace_name=workspace_name,auth=st.session_state["auth"],
                                                                      suffix="/g/api/graphql")
               #st.session_state["username"] = st.session_state["list_of_workspaces"]["data"]["session"]["user"]['name']
            except Exception as e:
                st.info(f"Psssword or login name are incorrect try again")
                st.session_state["username"]=""


                #raise(e)
                #print("DBG: auth",auth)
        except Exception as e:
            st.session_state["username"]=""
            st.error(e)
            #raise(e)
        
        #print(f"In Workspace Name: {workspace_name}. Number of processes: {len(list_processes_1)}")
        ### if st.session_state["username"] in st.secrets[
        ###     "passwords"
        ### ] and hmac.compare_digest(
        ###     st.session_state["password"],
        ###     st.secrets.passwords[st.session_state["username"]],
        ### ):
        ###     st.session_state["password_correct"] = True
        ###     del st.session_state["password"]  # Don't store the username or password.
        ###     del st.session_state["username"]
        if st.session_state["username"] !="":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    #print("DBG st.session_state.get password_correct", st.session_state.get("password_correct") )
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

# Successful Session ID definition
@st.cache_resource
def get_app_queue_success():
    return deque()

app_queue_success= get_app_queue_success()
if "success_session_id_list" not in st.session_state:
    app_queue_success.append({"success_session":f'{st.session_state.session_id}',
        "username":f"{st.session_state.username}",
        "session_creation_datetime": datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
                      })
    st.session_state.success_session_id_list=app_queue_success

# Main Streamlit app starts here
#st.write(f" Session UUID: {st.session_state.success_session_id_list}")
#st.write(st.session_state.auth)


if "list_of_workspaces" in st.session_state:
    st.write("Here List of ALL available workspaces...") 
    st.write (pd.DataFrame(st.session_state["list_of_workspaces"]["data"]["session"]["workspaces"]))

if "workspacename" in st.session_state and 'workspaceid' in st.session_state:
    workspace_name=st.session_state["workspacename"]
    workspace_id= st.session_state['workspaceid']
    st.write(f"Here List of available processes for selected workspace: {workspace_name} id: {workspace_id}") 
    list_processes =  POST_Signavio(query=q_list_processes,workspace_name=workspace_name,auth=st.session_state["auth"]) #['data']['subjects']
    st.write (pd.DataFrame(list_processes['data']['subjects']))
    #st.write(list_processes)
    st.text_input("Select Process id where we create LLM-based widgets ", "test00-1", key="process_id")
    # create_investigation
    #   "variables": {
    #    "subjectId": "test00-1",
    #    "investigation": {
    #        "name": "llm-based",
    #        "modelId": None,
    #        "revisionId": None,
    #        "viewId": "defaultview-2",
    #        "metrics": [],
    #        "access": "PRIVATE"
    #    }

def get_investigations_list():    
    #     "variables": {     "subjectId": "test00-1" },
    list_of_investigations["variables"]["subjectId"]=st.session_state["process_id"]
    investigations=POST_Signavio(query=list_of_investigations,workspace_name=workspace_name,auth=st.session_state["auth"])
    #st.write (investigations)
    st.session_state["investigations"]=investigations
    #st.text_input("Select Process view", "defaultview-02", key="process_view")
    #st.session_state.active_investigation=investigations["data"]["subject"]["investigations"][-1]
    #set_active_investigation(id=st.session_state.active_investigation)

def set_active_investigation(inv=""): 
    # Investigation details:
    # investigation_details = {
    #  "operationName": "Investigation",
    #   "variables": {
    # "investigationId": "llm-powered-28022024-180247-1",
    # "subjectId": "test00-1"
     # }
    if inv != "":
        active_investigation=inv
    else:
        active_investigation = st.session_state.active_investigation
    #print("IN set_active_investigation:",active_investigation )
    st.session_state.active_investigation=active_investigation
    try:
        query_investigation_details["variables"]["investigationId"]=active_investigation["id"]
        query_investigation_details["variables"]["subjectId"]=st.session_state["process_id"]
        #print("Q:",query_investigation_details)
        #print(query_investigation_details["variables"])
        active_investigation_details= POST_Signavio(query=query_investigation_details,workspace_name=workspace_name,auth=st.session_state["auth"])
        #print("qqqqq",active_investigation_details)
        st.session_state.active_investigation_details=active_investigation_details
    except Exception as e:
        st.error(f"Error in set_active_investigation {e}")


if "process_id" in st.session_state:
    #"operationName": "subjectViews",
    #"variables": {
    #    "subjectId": "demo01-1"
    #},
    process_id=st.session_state["process_id"]
    get_process_views["variables"]["subjectId"]=process_id
    list_processe_views =  POST_Signavio(query=get_process_views,workspace_name=workspace_name,auth=st.session_state["auth"]) #['data']['subjects']
    st.session_state.process_views=list_processe_views
    
    st.write(f"Here List of available investigations for process {process_id}:") 
    get_investigations_list()
    #st.write(st.session_state.process_views)

#get_active_investigations_list()

if "investigations" in st.session_state:
    st.write(pd.DataFrame(st.session_state.investigations["data"]["subject"]["investigations"]))
    #st.session_state.active_investigation= st.session_state.investigations["data"]["subject"]["investigations"][-1]


if "new_active_investigation_id" in st.session_state:
    new_id= st.session_state.new_active_investigation_id
    #print("new id ", new_id)
    try: 
        inv = [ inv for inv in st.session_state.investigations["data"]["subject"]["investigations"] if inv["id"] == new_id][0]
        st.session_state.active_investigation=inv
        #set_active_investigation(inv=inv)
    except Exception as e:
        #st.error(f"in new_active_investigation_id: {e}")
        st.session_state.active_investigation=st.session_state.investigations["data"]["subject"]["investigations"][-1]
        
if "active_investigation" not in st.session_state:
    st.session_state.active_investigation= st.session_state.investigations["data"]["subject"]["investigations"][-1]
    #active_investigation = st.session_state.active_investigation["id"]
    #print(active_investigation)

st.text_input("Select Investigation where we create LLM-based widgets ", st.session_state.active_investigation["id"] , key="new_active_investigation_id") 

if "active_investigation" in st.session_state:
    #if "active_investigation" in st.session_state:    
    #st.write(st.session_state["active_investigation"])
    st.write("Selected Investigation:")
    #print("got new invesigation", st.session_state.active_investigation)
    st.write(pd.DataFrame(st.session_state.active_investigation))




 

    
    #st.session_state.active_investigation= st.session_state.investigations["data"]["subject"]["investigations"][0]

#def get_active_investigation():
#    if "process_id" in st.session_state:
def create_new_investigation():
        try:
            stamp_name=datetime.now().strftime("(LLM-powered: %d/%m/%Y %H:%M:%S)")
            create_investigation["variables"]["investigation"]["name"]=stamp_name
            create_investigation["variables"]["subjectId"]=st.session_state["process_id"]
            
            # select latest process_views
            create_investigation["variables"]["investigation"]["viewId"]=st.session_state.process_views["data"]["subjectViews"][-1]["id"]
            investigation = POST_Signavio(query=create_investigation,workspace_name=st.session_state["workspacename"],auth=st.session_state["auth"])
            #print("NEW INV:",investigation)
            #st.write (investigation)
            #st.session_state["investigation"]=investigation
            #st.session_state.active_investigation=investigation["data"]["createInvestigation"]
            get_investigations_list()
            st.session_state.active_investigation=investigation["data"]["createInvestigation"]
            st.session_state.new_active_investigation_id=investigation["data"]["createInvestigation"]["id"]

            #
        except Exception as e:
            st.error(f"Error In create new investigation {e} ")
            #st.warning(st.session_state.active_investigation)


    #st.text_input("Select Active Investigation ", active_investigation, key="active_investigation")
st.button("Create new LLM-powered investigation", on_click=create_new_investigation)
#st.button("set_active_investigation", on_click=set_active_investigation)



get_investigations_list()
set_active_investigation()
    

# Process details
def signal_attributes():
  #  "variables": {
  #      "id": "defaultview-2",
  #      "subjectId": "test00-11"
  #  }
    #q_list_columns["variables"]["subjectId"]=st.session_state.active_investigation['id']
    #q_list_columns["variables"]["id"] = st.session_state.active_investigation['view']['id']
    q_list_columns["variables"]["id"]=st.session_state.active_investigation_details["data"]["investigation"]['view']['id']
    q_list_columns["variables"]["subjectId"]=st.session_state.active_investigation_details["data"]["investigation"]["id"]
    #print(q_list_columns["variables"]["id"],q_list_columns["variables"]["subjectId"])
    col_names = POST_Signavio(query=q_list_columns,workspace_name=workspace_name, auth=st.session_state["auth"])['data']['subjectView']['columns']
    schema_min=[el["name"] for el in col_names]
    query_events=f'SELECT DISTINCT(event_name) FROM FLATTEN("{q_list_columns["variables"]["id"]}")'
    #print("Query event_names", query_events)
    signal_endpoint = st.session_state["systeminstance"] + '/g/api/pi-graphql/signal'
    query_request = requests.post(
        signal_endpoint,
        cookies=st.session_state["auth"][workspace_name]["cookies"],
        headers=st.session_state["auth"][workspace_name]["headers"] ,
        json={'query': str(query_events) })
    events=query_request.json()    
    #print("khghgj",events)
    #logger.info(events)
    events_list=[item for row in events["data"] for item in row]
    out={"Attributes:":schema_min, 
         "EVENTS_NAMES": events_list}
    st.session_state['query_request_text'] = pd.DataFrame(col_names) #([out]) #(col_names) #'Initialization'
    st.session_state['query_list_of_events_text']=out

#st.button(label = "Attributes", on_click = signal_attributes)
       
st.sidebar.markdown(f"# Signavio connection details")
st.sidebar.write(f":red[USER:] {st.session_state.username}")
st.sidebar.write(f'Process: {st.session_state.active_investigation_details["data"]["subject"]["name"]}')
st.sidebar.write(f"Investigation: {st.session_state.active_investigation_details['data']['investigation']['name']}")
st.sidebar.write(f"view: {st.session_state.active_investigation_details['data']['investigation']['view']['id']}")
#st.sidebar.write(pd.DataFrame(st.session_state.active_investigation_details['data']['investigation']["widgets"]))
#st.write(st.session_state.active_investigation_details)
#st.write(f":red[USER:] {st.session_state.username}")

signal_attributes()
with st.chat_message('signal'):
    if st.session_state['query_list_of_events_text'] != "": # print schema for attributes only
        s = st.session_state['query_list_of_events_text']["Attributes:"]
        st.code(f'Attributes = {s}')
        st.code(f'EVENTS_NAMES= {st.session_state["query_list_of_events_text"]["EVENTS_NAMES"]}')#
        st.write(st.session_state['query_request_text']) 