## Setting up and running the streamlit app

To run the Streamlit app, follow these steps:

### 1. Setup the environment
Create a virtual environment and install the required dependencies using the requirementsUI.txt file:

```bash
python3 -m venv text2signal
source text2signal/bin/activate
pip install -r requirementsUI.txt
```

### 2. Set environment variables

Set the following environment variables in a .env file:

```dotenv
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=

AZURE_OPENAI_API_FT_KEY=
AZURE_OPENAI_FT_ENDPOINT=

# For UI
AZURE_OPENAI_BASE=
AZURE_OPENAI_KEY=
AZURE_OPENAI_MODEL=gpt-35-turbo-0613.ft-62a8ed1bf7594913a7041372c73227ef-text-to-signal
AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo-0613-text2signal-1epoch-lrm-5
```

### 3. Running the app

```bash
streamlit run notebooks/01_gui_signvio_signal_main_page.py
```
