# Text to SIGNAL (Signavio)

### Setup virtual environment

Initially, we attempted to set up the project using `poetry`. However, we encountered issues where the `poetry install` command was running indefinitely without completion, for unclear reasons.

If you prefer to use `poetry` and are able to resolve the issues, the `pyproject.toml` file is available in the repository.

However, due to the aforementioned issues, using `pip` with `requirements.txt` is the recommended approach.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Poetry setup

- Install [Poetry](https://python-poetry.org/docs/#installation).
- Get an Identity Token for <https://common.repositories.cloud.sap/>
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

[notebooks/README.md](notebooks/README.md)


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