# Text to SIGNAL (Signavio)

### Setup virtual environment

----
UPDATE: We have successfully configured the project to use Poetry for dependency management and package installation.
We recommend trying to use Poetry first for the most straightforward project setup experience.
To set up the project using Poetry, please check [Poetry setup](#Poetry setup)

----
Initially, we attempted to set up the project using `poetry`. However, we encountered issues where the `poetry install` command was running indefinitely without completion, for unclear reasons.

----
If you encounter any difficulties with Poetry, or simply prefer not to use it, we have provided a `requirements.txt` file as an alternative.
This allows you to use pip for installing dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### Prerequisites

- You need access to <https://signavio.jfrog.io>
- You can request this access via [CAM](https://spc.ondemand.com/sap/bc/webdynpro/a1sspc/cam_wd_central?item=request&profile=BPI%20Signavio%20JFrog%20Reader). This will grant you SSO to jfrog.
- You need to create an api key. Go to "Edit Profile" and click on "Generate an API Key".

### Poetry setup

- Install [Poetry](https://python-poetry.org/docs/#installation).
- Configure Poetry according to these [instructions](https://vigilant-waffle-9c99fbfa.pages.github.io/tools/pypi.html#setup), i.e. run `poetry config http-basic.signavio <username> <password>`. If your username contains a whitespace (`user name`), put it in quotes (`"user name"`).
- Repeat the similar procedure for the internet facing SAP repository. Get an Identity Token for <https://common.repositories.cloud.sap/>
    - Log in using “SAML SSO”
    - Navigate to your profile page
    - Click “Generate Identity Token” and give it a description and then click “Next”
    - Configure the Hyperspace repository with `poetry config http-basic.hyperspace <i/d-number> <identity-token>`.
- Repeat the previous step for <https://int.repositories.cloud.sap/> and set
    - `poetry config http-basic.internal-hyperspace <i/d-number> <identity-token>`
    - `poetry config http-basic.internal-hyperspace-deploy <i/d-number> <identity-token>`
- Run `poetry install` to install a virtual environment with all dependencies.
- Add libraries you are adding to the project (i.e., direct imports) to `pyproject.toml` and run `poetry lock --no-update` to update `poetry.lock` without changing the dependencies.
- Activate the virtual environment with `poetry shell`.

# Documentation on data collection, analysis, test/train split

[text2signal/README.md](text2signal/README.md)


1. Get signals from Different sources :[notebooks/1_Phase_GetSignalsFromSignavio.ipynb](notebooks/1_Phase_GetSignalsFromSignavio.ipynb)
2. Select samples which can be validated against real Signavio Process [notebooks/2_Phase_LoadSignalsFromCSV.ipynb](nonotebooks/2_Phase_LoadSignalsFromCSV.ipynb)
3. Augment data with LLM-generated description Create Train/test  JSONL [notebooks/3_Phase_LLMexperiments_withSubsetSignalsFromCSV.ipynb](notebooks/3_Phase_LLMexperiments_withSubsetSignalsFromCSV.ipynb)
   a. Requires : One needs to edit [**text2siganl/eval_reciprocal.py**](text2signal/eval_reciprocal.py)
    Insert above-created Folder name e.g.:  `inputfolder="training_data_3_with_validated_signals_2023-12-06T11_08_27"`
    Select LLM model `'deployment_id': "gpt-4"` to run

```py
# INPUT DIR is defined here
inputfolder="training_data_10_with_validated_signals_2023-12-05T13_12_55"
inputfolder="training_data_3_with_validated_signals_2023-12-06T11_08_27"
...
config = {
        'deployment_id': "gpt-4", #"gpt-4-32k",
        #'anthropic-claude-v2-100k', #"gpt-4-32k",
        # #'anthropic-claude-v1-100k',
...

$ python eval_reciprocal.py
```

### Refactored Notebooks
In our continuous effort to improve the project's usability and efficiency, we have refactored some of the initial notebooks.
These refactored notebooks are designed to replace the original first and second phases outlined in:
**1_Phase_GetSignalsFromSignavio.ipynb** and **2_Phase_LoadSignalsFromCSV.ipynb**.

[notebooks/Refactored_1_Phase_GetSignalsFromSignavio.ipynb](notebooks/Refactored_1_Phase_GetSignalsFromSignavio.ipynb)
[notebooks/Refactored_2_Phase_LoadSignalsFromCSV.ipynb](notebooks/Refactored_2_Phase_LoadSignalsFromCSV.ipynb)
