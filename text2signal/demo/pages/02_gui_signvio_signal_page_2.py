"""
StreamLit example was taken from
https://gist.github.com/paolosalvatori/478944506a809313de162759442df8c0
# This sample is based on the following article:
# - https://levelup.gitconnected.com/its-time-to-create-a-private-chatgpt-for-yourself-today-6503649e7bb6
#
# Use the following command to run the app:
# - streamlit run notebooks/gui_signvio_signal.py
# Create .env file with following variables:
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import chromadb
import openai

# Signavio
import pandas as pd
import requests
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import dotenv_values, load_dotenv
from openai import AzureOpenAI
from signavio_lib import (
    POST_Signavio,
    q_list_columns,
    query_to_api_signal,
    query_to_api_table,
)

# RAG
from tqdm import tqdm

# Second page
if "auth" not in st.session_state:
    st.write("Please re-Authenticate")
    if st.button("Home"):
        st.switch_page("01_gui_signvio_signal_main_page.py")
    st.stop()


# Load environment variables from .env file
env_path = ".env"
if os.path.exists(env_path):
    load_dotenv(override=True)
    config = dotenv_values(env_path)
    print(f"Config: {config}")


# Read environment variables

signavio_assistant_profile = """
You are SIGNAL assistant, a part of SAP Signavio's Process Intelligence Suite. \nSIGNAL stands for Signavio Analytics Language. \nSIGNAL is a dialect of SQL.\nYour goal is to help users craft SIGNAL queries and understand the SIGNAL language better
"""


title = os.environ.get("TITLE", "Signavio Signal ChatBot SandBox")
text_input_label = os.environ.get("TEXT_INPUT_LABEL", "Provide your NLP description for Signal:")
image_file_name = os.environ.get("IMAGE_FILE_NAME", "./signavioPI.png")
image_width = int(os.environ.get("IMAGE_WIDTH", 220))
temperature = float(os.environ.get("TEMPERATURE", 0.0))
system = os.environ.get("SYSTEM", signavio_assistant_profile)
api_base = os.getenv("AZURE_OPENAI_BASE")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_type = os.environ.get("AZURE_OPENAI_TYPE", "azure")
api_version = os.environ.get("AZURE_OPENAI_VERSION", "2023-05-15")
engine = os.getenv("AZURE_OPENAI_DEPLOYMENT")
model = os.getenv("AZURE_OPENAI_MODEL")

# Working
api_base = os.getenv("AZURE_OPENAI_BASE")
api_version = os.environ.get("AZURE_OPENAI_VERSION", "2023-12-01-preview")
api_key = os.getenv("AZURE_OPENAI_KEY")
model = "gpt-35-turbo-0613-text2signal-1epoch-lrm-5"  # 1 epoch lr*5
azure_endpoint = os.getenv("AZURE_OPENAI_FT_ENDPOINT")
# Signavio


system_instance = st.session_state["systeminstance"]  # 'https://editor.signavio.com'
workspace_id = st.session_state["workspaceid"]  # 'b0f07deabd3140aea5344baa686e0d84' # workspace Process AI
workspace_name = st.session_state["workspacename"]  # "Process AI"

# Configure a logger
logging.basicConfig(
    stream=sys.stdout, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Log variables


logger.info(f"title: {title}")
logger.info(f"text_input_label: {text_input_label}")
logger.info(f"image_file_name: {image_file_name}")
logger.info(f"image_width: {image_width}")
logger.info(f"temperature: {temperature}")
logger.info(f"system: {system}")
logger.info(f"api_base: {api_base}")
logger.info(f"api_key: {api_key}")
logger.info(f"api_type: {api_type}")
logger.info(f"api_version: {api_version}")
logger.info(f"engine: {engine}")
logger.info(f"model: {model}")


# Configure OpenAI
openai.api_type = api_type
# openai.api_version = api_version
# openai.api_base = api_base

# Set default Azure credential
# default_credential = DefaultAzureCredential() if openai.api_type == "azure_ad" else None
# token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")


logger.info(f"AzureOpenAI azure_endpoint: {azure_endpoint}")
logger.info(f"AzureOpenAI api_key: {api_key}")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=api_key,  # azure_ad_token_provider=token_provider,
)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, api_type="azure", api_base=api_base, api_version=api_version, model_name="text-embedding-ada-002"
)

# Chromadb
file_path = "text2signal_train_5715.jsonl"
# file_path='notebooks/probe.jsonl'
path_db = "chromadb_embeddings"
collection_name = "my_collection"
collection_name = "collection_5715_train_set"

if not Path(path_db).is_dir():
    st.error("Check path to DB (l:142) or Click I know what I am doing")
    if st.button("I know what I am doing"):
        st.write("Start constructing ChromaDB for RAG - takes 20 mins...")
    else:
        st.stop()

    client_db = chromadb.PersistentClient(path=path_db)
    client_db.heartbeat()
    dataset_in = []
    with open(file_path) as f:
        for line in f:
            dataset_in.append(json.loads(line))
    # DB
    try:
        client_db.delete_collection(name=collection_name)
        st.session_state.messages = []
    except:
        print("Hopefully you'll never see this error.")

    collection = client_db.create_collection(name=collection_name, embedding_function=openai_ef)
    data = []
    id = 1
    # {"filename": "ffdddc22bd",
    # "split": "train",
    # "name": "Number of open overdue invoices",
    # "query": "SELECT\n  COUNT(case_id) FILTER (\n    WHERE\n      (\"Last Inv. Item Clearing Date\" IS NULL)\n      AND \"Last Inv. Item Due Date\" < DATE_TRUNC('day', NOW())\n  )\nFROM\n  \"defaultview-225\"",
    # "llm_name": "signal_description_llama2-70b-chat-hf",
    # "description_llm": "Count the number of case_id where Last Inv. Item Clearing Date is null and Last Inv. Item Due Date is before the current day."}

    SIGNAL = "query"
    SIGNAL_NAME = "name"
    SIGNAL_FILENAME = "filename"
    SIGNAL_LLM_NAME = "llm_name"
    CONTENT = "description_llm"

    for dict in tqdm(dataset_in):
        content = dict.get(CONTENT, "")
        signal = dict.get(SIGNAL, "")
        signal_name = dict.get(SIGNAL_NAME, "")
        signal_filename = dict.get(SIGNAL_FILENAME, "")
        signal_llm_name = dict.get(SIGNAL_LLM_NAME, "")

        content_metadata = {
            SIGNAL: signal,
            SIGNAL_NAME: signal_name,
            SIGNAL_FILENAME: signal_filename,
            SIGNAL_LLM_NAME: signal_llm_name,
        }

        collection.add(documents=[content], metadatas=[content_metadata], ids=[str(id)])
        id += 1
    logger.info(f" {collection.count()} <------------ NEW DB")  # ,collection.peek())

if Path(path_db).is_dir():
    # client = chromadb.Client(Settings(is_persistent=True,
    #                                persist_directory= <PERSIST_DIR_NAME>,
    #                            ))
    client_db = chromadb.PersistentClient(path=path_db)
    client_db.heartbeat()
    collection = client_db.get_collection(name=collection_name, embedding_function=openai_ef)
    # collection = client_db.get_collection("my_collection")

    # logger.info(collection.count(),"DB from files:",collection.peek())
    logger.info(f" {collection.count()} <------------ DB from dir: {path_db}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Authenicate Signavio

# user_name = os.environ.get('MY_SIGNAVIO_NAME','alexey.streltsov@sap.com') # username
# pw = os.environ.get('MY_SIGNAVIO_PASSWORD', None) # Signavio password
# system_instance = 'https://editor.signavio.com'
# workspace_id = 'b0f07deabd3140aea5344baa686e0d84' # workspace Process AI
# workspace_name="Process AI"
# logger.info(f"Your are: {user_name}")
# auth = credentials_actualization(system_instance, workspace_id, user_name, pw, workspace_name=workspace_name)
auth = st.session_state["auth"]
logger.info(f"Signavio Auth: {system_instance} {workspace_id} {auth}")

# Authenticate to Azure OpenAI
### if openai.api_type == "azure":
###   openai.api_key = api_key
### elif openai.api_type == "azure_ad":
###   openai_token = default_credential.get_token("https://cognitiveservices.azure.com/.default")
###   openai.api_key = openai_token.token
###   if 'openai_token' not in st.session_state:
###     st.session_state['openai_token'] = openai_token
### else:
###   logger.error("Invalid API type. Please set the AZURE_OPENAI_TYPE environment variable to azure or azure_ad.")
###   #raise ValueError("Invalid API type. Please set the AZURE_OPENAI_TYPE environment variable to azure or azure_ad.")

st.set_page_config(page_title="Signavio ChatBot-Lab", layout="centered")

# Customize Streamlit UI using CSS
style = """
<style>

.big-font {
    font-size:20px !important;
}
div.stButton > button:first-child {
    background-color: #eb5424;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: none;
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
    width: 300 px;
    height: 42px;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:first-child:hover {
    transform: translateY(-3px);
    box-shadow: 0 1rem 2rem rgba(0,0,0,0.15);
}

div.stButton > button:first-child:active {
    transform: translateY(-1px);
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
}

div.stButton > button:focus:not(:focus-visible) {
    color: #FFFFFF;
}

@media only screen and (min-width: 768px) {
  /* For desktop: */
  div {
      font-family: 'Roboto', sans-serif;
  }

  div.stButton > button:first-child {
      background-color: #eb5424;
      color: white;
      font-size: 20px;
      font-weight: bold;
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      border: none;
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
      width: 300 px;
      height: 42px;
      transition: all 0.2s ease-in-out;
      position: relative;
      bottom: -32px;
      right: 0px;
  }

  div.stButton > button:first-child:hover {
      transform: translateY(-3px);
      box-shadow: 0 1rem 2rem rgba(0,0,0,0.15);
  }

  div.stButton > button:first-child:active {
      transform: translateY(-1px);
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
  }

  div.stButton > button:focus:not(:focus-visible) {
      color: #FFFFFF;
  }

  input {
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      border: none;
      box-shadow: 0 0.5rem 1rem rgba(0,0,1,0.15);
      transition: all 0.2s ease-in-out;
      height: 40px;
  }
}

.st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
"""
# style=""
st.sidebar.header("Signavio LLM SandBox")
st.markdown(style, unsafe_allow_html=True)


# Initialize Streamlit session state
if "prompts" not in st.session_state:
    st.session_state["prompts"] = [{"role": "system", "content": system}]

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "usage" not in st.session_state:
    st.session_state["usage"] = []

if "current_cost" not in st.session_state:
    st.session_state["current_cost"] = "0.0"
if "current_tokens" not in st.session_state:
    st.session_state["current_tokens"] = "0"

if "widget_summary" not in st.session_state:
    st.session_state["widget_summary"] = []

if "query_request_text" not in st.session_state:
    st.session_state["query_request_text"] = ""

if "query_request" not in st.session_state:
    st.session_state["query_request"] = ""
if "query_list_of_events_text" not in st.session_state:
    st.session_state["query_list_of_events_text"] = ""
if "query" not in st.session_state:
    st.session_state["query"] = 'SELECT DISTINCT(event_name) FROM FLATTEN("defaultview-2")'
    # pd.DataFrame(col_names) #'Initialization'

# RAG
if "rag_results" not in st.session_state:
    st.session_state["rag_results"] = []

# Refresh the OpenAI security token every 45 minutes
# def refresh_openai_token():
#  if st.session_state['openai_token'].expires_on < int(time.time()) - 45 * 60:
#      st.session_state['openai_token'] = default_credential.get_token("https://cognitiveservices.azure.com/.default")
#      openai.api_key = st.session_state['openai_token'].token


def rag_prompt_query():
    prompt = st.session_state["rag_prompt_query"]
    # query ChromaDB based on your prompt, taking the top 5 most relevant result. These results are ordered by similarity.
    q = collection.query(
        query_texts=[prompt],
        n_results=5,
    )
    st.session_state["rag_results"] = q  # ["documents"][0]
    logger.info(f"RAG: {q.keys()}")


def signal_change():
    # Avoid handling the event twice when clicking the Send button
    signal_input = st.session_state["query"]
    logger.info(f"Input Signal: {signal_input}")
    signal_endpoint = system_instance + "/g/api/pi-graphql/signal"
    query_request = requests.post(
        signal_endpoint,
        cookies=auth[workspace_name]["cookies"],
        headers=auth[workspace_name]["headers"],
        json={"query": str(signal_input)},
    )
    # st.session_state['query']
    # logger.info(f"{query_request.json()}")
    st.session_state["query_request"] = f"{query_request.json()}"
    # st.session_state['query_request_text']= ""


#     col_names = POST_Signavio(query=q_list_columns,workspace_name=workspace_name, auth=auth)['data']['subjectView']['columns']
#          st.button(label = "Get Attributes", on_click = signal_attributes
def signal_attributes():
    #  "variables": {
    #      "id": "defaultview-2",
    #      "subjectId": "test00-11"
    #  }
    # q_list_columns["variables"]["subjectId"]=st.session_state.active_investigation['id']
    # q_list_columns["variables"]["id"] = st.session_state.active_investigation['view']['id']
    q_list_columns["variables"]["id"] = st.session_state.active_investigation_details["data"]["investigation"]["view"][
        "id"
    ]
    q_list_columns["variables"]["subjectId"] = st.session_state.active_investigation_details["data"]["investigation"][
        "id"
    ]
    col_names = POST_Signavio(query=q_list_columns, workspace_name=workspace_name, auth=auth)["data"]["subjectView"][
        "columns"
    ]
    schema_min = [el["name"] for el in col_names]
    query_events = 'SELECT DISTINCT(event_name) FROM FLATTEN("defaultview-2")'
    signal_endpoint = system_instance + "/g/api/pi-graphql/signal"
    query_request = requests.post(
        signal_endpoint,
        cookies=auth[workspace_name]["cookies"],
        headers=auth[workspace_name]["headers"],
        json={"query": str(query_events)},
    )
    events = query_request.json()
    # logger.info(events)
    events_list = [item for row in events["data"] for item in row]
    out = {"Attributes:": schema_min, "EVENTS_NAMES": events_list}
    st.session_state["query_request_text"] = pd.DataFrame(col_names)  # ([out]) #(col_names) #'Initialization'
    st.session_state["query_list_of_events_text"] = out


def arr_dimen(a):
    return [len(a)] + arr_dimen(a[0]) if (type(a) == list) else []


def signal_widget_prepare():
    # stamp=datetime.now().strftime("(LLM-created: %d/%m/%Y %H:%M:%S)")
    signal = st.session_state["widget_signal"]
    prompt_title = f"""
  Describe this query by one short sentence
  signal query:
  {signal}
  """
    prompt_description = f"""
  Describe in details how this query works
  signal query:
  {signal}
  """
    message_title, usage = generate_response(prompt_title)
    message_description, usage = generate_response(prompt_description)
    st.session_state["widget_title"] = message_title
    st.session_state["widget_description"] = message_description


def signal_widget_deploy():
    # Timestamp and Logo
    stamp = datetime.now().strftime("(LLM-created: %d/%m/%Y %H:%M:%S)")
    title = st.session_state["widget_title"]
    signal = st.session_state["widget_signal"]
    description = st.session_state["widget_description"]
    r = {"creation_datetime": stamp, "title": title, "signal": signal, "description": description}
    signal_endpoint = system_instance + "/g/api/pi-graphql/signal"
    query_request = requests.post(
        signal_endpoint,
        cookies=auth[workspace_name]["cookies"],
        headers=auth[workspace_name]["headers"],
        json={"query": str(signal)},
    )
    r["validation"] = query_request.json()
    try:
        # if r["validation"]['header'][0]['dataType'] !=  'NUMBER':
        dim_data = arr_dimen(r["validation"]["data"])
        if dim_data[0] != 1:
            widget_template = query_to_api_table
            r["widget_type"] = "Table"
        else:
            widget_template = query_to_api_signal
            r["widget_type"] = "Single Number"
        logging.info(f'Widget type: {dim_data} {r["widget_type"]}')
        # replace parts
        # SELECT count(event_name) AS "# Events", event_name FROM FLATTEN("defaultview-2") ORDER BY 1 DESC LIMIT 20
        widget_template["variables"]["widget"]["dataSource"]["query"] = signal
        widget_template["variables"]["widget"]["name"] = f"{stamp} : {title}"
        widget_template["variables"]["widget"]["description"] = f"{description}"
        r["widget_template"] = widget_template

        # widget_template["variables"]["id"] = f"{st.session_state.active_investigation['id']}"
        widget_template["variables"]["id"] = st.session_state.active_investigation_details["data"]["investigation"][
            "rootWidget"
        ]
        res = POST_Signavio(query=widget_template, workspace_name=workspace_name, auth=auth)
        r["ok"] = res  # .json()
        st.session_state["widget_summary"] = r  # json.dumps(r)

    except Exception as e:
        st.session_state["widget_title"] = "Error trying to create a widget"
        st.session_state["widget_description"] = f"Error: {e}"
        st.session_state["widget_summary"] = r["validation"]
        logging.error(f'Widget creation error: {e} {r["validation"]}')

    #


# Send user prompt to Azure OpenAI
def generate_response(prompt):
    try:
        st.session_state["prompts"].append({"role": "user", "content": prompt})

        # if openai.api_type == "azure_ad":
        #  refresh_openai_token()

        completion = client.chat.completions.create(
            model=model,
            messages=st.session_state["prompts"],
            temperature=temperature,
        )
        # logger.info(f"{st.session_state['prompts']} STR: {completion}")

        message = completion.choices[0].message.content

        usage = json.loads(completion.model_dump_json(indent=2))["usage"]
        # logger.info(f"USAGE: {usage}")

        return message, usage
    except Exception as e:
        logging.exception(f"Exception in generate_response: {e}")


# Reset Streamlit session state to start a new chat from scratch
def new_click():
    st.session_state["prompts"] = [{"role": "system", "content": system}]
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["usage"] = []
    st.session_state["rag_results"] = []
    st.session_state["llm_prompt"] = ""
    st.session_state["current_cost"] = "0.0"
    st.session_state["current_tokens"] = "0"

    st.session_state["completion_tokens"] = ""
    st.session_state["prompt_tokens"] = ""
    st.session_state["total_tokens"] = ""
    st.session_state["widget_summary"] = ""
    st.session_state["widget_title"] = "paste answer from LLM: Describe query by one short sentence"
    st.session_state["widget_description"] = ""

    # Validation
    st.session_state["query_request"] = ""
    st.session_state["query_request_text"] = ""
    st.session_state["query"] = ""


# Handle on_change event for llm_prompt input
def llm_prompt():
    # Avoid handling the event twice when clicking the Send button
    chat_input = st.session_state["llm_prompt"]
    # st.session_state['llm_prompt'] = ""
    # if (chat_input == '' or
    #    (len(st.session_state['past']) > 0 and chat_input == st.session_state['past'][-1])):
    #  return

    # Generate response invoking Azure OpenAI LLM
    if chat_input != "":
        output, usage = generate_response(chat_input)

        # store the output
        st.session_state["past"].append(chat_input)
        st.session_state["generated"].append(output)
        st.session_state["prompts"].append({"role": "assistant", "content": output})

        st.session_state["usage"].append(usage)
        st.session_state["completion_tokens"] = usage["completion_tokens"]
        st.session_state["prompt_tokens"] = usage["prompt_tokens"]
        st.session_state["total_tokens"] = usage["total_tokens"]
        ct, pt, tt = 0, 0, 0
        for el in st.session_state["usage"]:
            ct += el["completion_tokens"]
            pt += el["prompt_tokens"]
            tt += el["total_tokens"]
        st.session_state["current_cost"] = f" {round((int(pt)*0.0030 + int(ct)*0.0060)/1024,5)}"
        st.session_state["current_tokens"] = f"{round(tt/1024,5)}"
        logger.info(f"---------------> Tokens. Completion {ct}. Prompt: {pt}. Total:{tt} ")
        # input $0.0030 / 1K tokens  output $0.0060 / 1K tokens
        # logger.info(f"Cost: {st.session_state['current_cost']}")


# --------------- > UI
# Create a 3-column layout. Note: Streamlit columns do not properly render on mobile devices.
# For more information, see https://github.com/streamlit/streamlit/issues/5003
col1, col2, col_price, col_tokens = st.columns([3, 2, 2, 2])

# Display the robot image
with col1:
    st.image(image=image_file_name, width=image_width)

# Display the title
with col2:
    # st.title(title)
    st.write(f'<p class="big-font"> {title} </p>', unsafe_allow_html=True)

with col_price:
    text_price = st.metric(label="Total Price [$]:", value=st.session_state["current_cost"])
with col_tokens:
    text_tokens = st.metric(label="Total Tokens [k]:", value=st.session_state["current_tokens"])
    logger.info(float(st.session_state["current_tokens"]))
    if float(st.session_state["current_tokens"]) > 4.02:
        st.error(
            f'You reach prompt length limit {round(float(st.session_state["current_tokens"]),2)} > 4k. Renew session!'
        )


# Create a 3-column layout. Note: Streamlit columns do not properly render on mobile devices.
# For more information, see https://github.com/streamlit/streamlit/issues/5003
col3, col4, col5 = st.columns([7, 1, 1])


# Create text input in column 1
with col3:
    user_input = st.text_area(text_input_label, key="llm_prompt")  # , on_change = llm_prompt)

# Create send button in column 2
with col4:
    st.button(label="Send", on_click=llm_prompt)

# Create new button in column 3
with col5:
    st.button(label="New", on_click=new_click)


# Display the chat history in two separate tabs
# - normal: display the chat history as a list of messages using the streamlit_chat message() function
# - rich: display the chat history as a list of messages using the Streamlit markdown() function
# if st.session_state['generated']:
# tab1, tab2, tab3 = st.tabs(["normal", "rich","Signal"])
tab1, tab2, tab3, tab4 = st.tabs(["FT gpt3.5-turbo LLM", "RAG", "Signal Queries", "Create Signal Widget"])
with tab1:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        with st.chat_message("user"):  # , avatar="🤖"):
            st.write(st.session_state["past"][i])
        with st.chat_message("assistant"):  # ,avatar="🤖"): #, avatar="🦖"):
            st.write(st.session_state["generated"][i])
#  RAG
with tab2:
    col_rag_prompt, col_rag_button = st.columns([6, 1])
    with col_rag_prompt:
        col_rag_prompt_input = st.text_area(
            "Type/Paste Signal description here:", key="rag_prompt_query", on_change=rag_prompt_query, height=75
        )
    with col_rag_button:
        st.button(label="RAG", on_click=rag_prompt_query)
    res = st.session_state["rag_results"]
    if isinstance(res, dict) and "ids" in res.keys():
        ids = res["ids"][0]
        distances = res["distances"][0]
        metadatas = res["metadatas"][0]
        documents = res["documents"][0]
        for i in range(len(ids)):
            with st.chat_message("user"):  # , avatar="🤖"):
                st.write(documents[i])
                # with st.chat_message('distance'): #, avatar="🤖"):
                st.write(f"Distance: {distances[i]}")
                # with st.chat_message('query'): #, avatar="🤖"):
                st.write(metadatas[i])
# Signavio
with tab3:
    # Add Signavio part
    col_signal_validation, col_signal_validation_button = st.columns([6, 1])
    with col_signal_validation:
        user_input = st.text_area(
            "Paste/Type Signal here for API validation:", key="query", on_change=signal_change, height=35
        )
    with col_signal_validation_button:
        st.button(label="Validate", on_click=signal_change)
        #     col_names = POST_Signavio(query=q_list_columns,workspace_name=workspace_name, auth=auth)['data']['subjectView']['columns']
        st.button(label="Attributes", on_click=signal_attributes)

    with st.chat_message("user"):
        st.write(st.session_state["query_request"])
    with st.chat_message("signal"):
        if st.session_state["query_list_of_events_text"] != "":  # print schema for attributes only
            st.write(st.session_state["query_list_of_events_text"])  #
            st.write(st.session_state["query_request_text"])

    # message(st.session_state['query_request'])
    with st.chat_message("ai"):  # , avatar="LLM"): # avatar=st.image("docs/deer-head.svg")):
        st.write(st.session_state["query"])
    # message(st.session_state['query'], avatar_style = "bottts", seed = "Fluffy")
# Widget
with tab4:
    w_name, w_button = st.columns([6, 1])
    with w_name:
        signal = st.text_input("Widget Signal", "Paste Validated Signal query here ", key="widget_signal")
        st.write("The current Widget Signal:", signal)
        title = st.text_input("Widget name", key="widget_title")
        st.write("The current Widget title:", title)
        description = st.text_input(
            "Widget description", "Paste here answer from LLM: Describe how this query works", key="widget_description"
        )
        st.write("The current Widget description:", description)
        with st.chat_message("user"):
            st.write(st.session_state["widget_summary"])
with w_button:
    st.button(label="Create", on_click=signal_widget_prepare)
    st.button(label="Deploy", on_click=signal_widget_deploy)
    # st.text_area("Widget Details:", key = "widget_summary", height=35)
