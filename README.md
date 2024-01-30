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
