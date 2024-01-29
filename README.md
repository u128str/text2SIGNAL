# template for python-based projects at process.ai

# Introduction
- describe the project, also from a non-technical perspective
- describe the goal of the project
- describe if the repo is rather explanatory or meant to become production code

# Setup
- describe how to set up the project
- this includes how to install tools, how to set up the environment, how to install dependencies

```
    ├── package_name
    │   └── module_1
    │       ├── schemas.py
    │       ├── logic_of_module_1.py
    │       └── ...
    │   └── module_2
    │       ├── schemas.py
    │       ├── logic_of_module_2.py
    │       └── ...
    │   └── tests
    │       ├── unit
    │       │   ├── test_module_1.py
    │       │   └── ...
    │       │   integration
    │       │   ├── test_module_1.py
    │       │   └── ...
    |   └── main.py
    |   └── pyproject.toml
    
```