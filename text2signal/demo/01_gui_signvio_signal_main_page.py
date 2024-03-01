# This is first starting page for Text2Signal SandBox UI
# $ streamlit run notebooks/01_gui_signvio_signal_main_page.py

import streamlit as st

st.markdown("# Login page ")
st.sidebar.markdown("# Login page")

from datetime import datetime

import pandas as pd

# Signavio
from signavio_lib import (
    POST_Signavio,
    create_investigation,
    credentials_actualization,
    get_process_views,
    list_of_investigations,
    q_list_processes,
    query_investigation_details,
)


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information."""
        with st.form("Credentials"):
            st.text_input("Username", "alexey.streltsov@sap.com", key="username")
            st.text_input("Password", type="password", key="password")
            # system_instance = 'https://editor.signavio.com'
            # workspace_id = 'b0f07deabd3140aea5344baa686e0d84' # workspace Process AI
            # workspace_name="Process AI"
            st.text_input("system_instance", "https://editor.signavio.com", key="system_instance")
            st.text_input("workspace_id", "b0f07deabd3140aea5344baa686e0d84", key="workspace_id")
            st.text_input("workspace_name", "Process AI", key="workspace_name")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # system_instance=st.session_state["system_instance"]
        # workspace_id= st.session_state["workspace_id"]
        # workspace_name= st.session_state["workspace_name"]
        #
        # username= st.session_state["username"]
        # pw=st.session_state["password"]
        try:
            workspace_name = st.session_state["workspace_name"]
            auth = credentials_actualization(
                st.session_state["system_instance"],
                st.session_state["workspace_id"],
                st.session_state["username"],
                st.session_state["password"],
                workspace_name=workspace_name,
            )
            st.session_state.auth = auth
            st.session_state["workspacename"] = st.session_state["workspace_name"]
            st.session_state["systeminstance"] = st.session_state["system_instance"]
            st.session_state["workspaceid"] = st.session_state["workspace_id"]
            # print(auth)
        except Exception as e:
            st.session_state["username"] = ""
            raise (e)

        # print(f"In Workspace Name: {workspace_name}. Number of processes: {len(list_processes_1)}")
        ### if st.session_state["username"] in st.secrets[
        ###     "passwords"
        ### ] and hmac.compare_digest(
        ###     st.session_state["password"],
        ###     st.secrets.passwords[st.session_state["username"]],
        ### ):
        ###     st.session_state["password_correct"] = True
        ###     del st.session_state["password"]  # Don't store the username or password.
        ###     del st.session_state["username"]
        if st.session_state["username"] != "":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    print("DBG st.session_state.get password_correct", st.session_state.get("password_correct"))
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

# Main Streamlit app starts here
st.write("Here List of available processes...")


if "workspacename" in st.session_state:
    workspace_name = st.session_state["workspacename"]
    list_processes = POST_Signavio(
        query=q_list_processes, workspace_name=workspace_name, auth=st.session_state["auth"]
    )  # ['data']['subjects']
    st.write(pd.DataFrame(list_processes["data"]["subjects"]))
    st.write(list_processes)
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

if "process_id" in st.session_state:
    # "operationName": "subjectViews",
    # "variables": {
    #    "subjectId": "demo01-1"
    # },
    get_process_views["variables"]["subjectId"] = st.session_state["process_id"]
    list_processe_views = POST_Signavio(
        query=get_process_views, workspace_name=workspace_name, auth=st.session_state["auth"]
    )  # ['data']['subjects']
    st.session_state.process_views = list_processe_views
    st.write(st.session_state.process_views)


# def get_active_investigation():
#    if "process_id" in st.session_state:
def create_new_investigation():
    stamp_name = datetime.now().strftime("(LLM-powered: %d/%m/%Y %H:%M:%S)")
    create_investigation["variables"]["investigation"]["name"] = stamp_name
    create_investigation["variables"]["subjectId"] = st.session_state["process_id"]

    # select latest
    create_investigation["variables"]["investigation"]["viewId"] = st.session_state.process_views["data"][
        "subjectViews"
    ][-1]["id"]

    investigation = POST_Signavio(
        query=create_investigation, workspace_name=st.session_state["workspacename"], auth=st.session_state["auth"]
    )
    print(investigation)
    st.write(investigation)
    st.session_state["investigation"] = investigation
    st.session_state.active_investigation = investigation["data"]["createInvestigation"]
    get_active_investigations_list()
    set_active_investigation()


def get_active_investigations_list():
    #     "variables": {     "subjectId": "test00-1" },
    list_of_investigations["variables"]["subjectId"] = st.session_state["process_id"]
    investigations = POST_Signavio(
        query=list_of_investigations, workspace_name=workspace_name, auth=st.session_state["auth"]
    )
    st.write(investigations)
    st.session_state["investigations"] = investigations
    # st.text_input("Select Process view", "defaultview-02", key="process_view")
    st.session_state.active_investigation = investigations["data"]["subject"]["investigations"][-1]
    set_active_investigation()


def set_active_investigation():
    # Investigation details:
    # investigation_details = {
    #  "operationName": "Investigation",
    #   "variables": {
    # "investigationId": "llm-powered-28022024-180247-1",
    # "subjectId": "test00-1"
    # }
    query_investigation_details["variables"]["investigationId"] = st.session_state.active_investigation["id"]
    query_investigation_details["variables"]["subjectId"] = st.session_state["process_id"]
    # print(query_investigation_details["variables"])
    active_investigation_details = POST_Signavio(
        query=query_investigation_details, workspace_name=workspace_name, auth=st.session_state["auth"]
    )
    # print("hgkjhjh",active_investigation_details)
    st.session_state.active_investigation_details = active_investigation_details

    # st.text_input("Select Active Investigation ", active_investigation, key="active_investigation")


if "investigations" in st.session_state:
    st.write(pd.DataFrame(st.session_state.investigations["data"]["subject"]["investigations"]))

if "active_investigation" in st.session_state:
    st.write(st.session_state["active_investigation"])
    st.write("Selected Investigation:")
    st.write(pd.DataFrame(st.session_state.active_investigation))

if "active_investigation_details" in st.session_state:
    st.write(st.session_state.active_investigation_details)

st.button("List of Investigations", on_click=get_active_investigations_list)

st.button("Create new LLM-powered investigation", on_click=create_new_investigation)
st.button("set_active_investigation", on_click=set_active_investigation)
