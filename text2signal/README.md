
# Signavio Text2Signal PoC use-cаsе

Here are the links to relevant sources:

- Ask PoC with 9 samples of Signal Query ( kudo to Wheeler, Mark):  [here is a link to the package](https://sap.sharepoint.com/:u:/t/OttoKraken/EdXVVYaUzHpOvK0H_EOg8gQBmMwQzbakcNzqvg3aDbynvg?email=alexey.streltsov%40sap.com&e=VLrlE3)
  - [AskData SIGNAL POC - Use Cases and Query Set.txt](AskData_SIGNAL_POC-Use_Cases_and_Query_Set.txt)
- Git to gpt4-32k doc use-case server solution: [signal-query-assistant](https://github.com/signavio/signal-query-assistant.git)
- Link to Signavio UI (registration required, contact Andreas Gerber BPI Product): [UI](https://editor.signavio.com/g/statics/pi/areas)
- Link to SIGNAL documentation in HTML format: [SIGNAL DOCS](https://help.sap.com/docs/signavio-process-intelligence/signal-guide/syntax)
- Link to the notebook where API calls are defined: [API Calls example](https://github.com/signavio/bpmn2constraints/blob/main/tutorial/tutorial.ipynb)  
- Process Manager: [API endpoints documentation](https://help.sap.com/doc/sap-signavio-process-manager-api-documentation/SHIP/en-US/rest_api/index.html)

## Data sources:

- 9 samples **AskData** contact: mark.wheeler@sap.com  [AskData SIGNAL POC - Use Cases and Query Set.txt](data/AskData_SIGNAL_POC-Use_Cases_and_Query_Set.txt)
- 488 (481) samples **PEX** contact: mark.wheeler@sap.com [data/metric_definitions](data/metric_definitions) origin: [external git](https://github.com/signavio/pex-metric-library/tree/main/metric_definitions)
- 942 (645) samples **PIA** contact: mark.wheeler@sap.com [data/pia-dashboards](data/pia-dashboards) origin: [external git](https://github.tools.sap/SAP-Signavio-PIA/pia-dashboards)

- 67 (39) samples **Plug and Gain OPAL** contact: maximilian.hoelzig@sap.com [data/OPAL](data/OPAL)

- 3897 (610) samples **PnG** contact: maximilian.hoelzig@sap.com [data/ContentPackage_November2023](data/ContentPackage_November2023) data from [this git](https://github.tools.sap/business-process-insights/bpi-python-tools/tree/main/content/ContentPackage_November2023)
- ?under discussion? samples **SAP Signavio Suite Power Challenge** contact: sarah.roessler@sap.com

- 4266 (1399) samples **Solutions Demo Workspace** contact: alexandra.dang@sap.com  [data/Frome_API/Solutions Demo Workspace](data/From_API/Solutions%20Demo%20Workspace/) [API](https://editor.signavio.com/g/statics/pi/areas/etl-220/processes/00demo---sap-s4h-itp-dashboar-1/dashboards)
  - (10 November) 33 processes with 46 investigations, 167 dashboards
  - (14 November)  N processes 29. Dashboards: 84. Investigations 34.
  - (17 November) Workspace name: Solutions Demo Workspace: N processes 31. Dashboards: 110. Investigations 43. Metrics: 658

<details>
<summary><i> How to get data in Pandas DataFame </i></summary>

Run notebooks:
[notebooks/LoadSignalsFromCSV.ipynb](notebooks/LoadSignalsFromCSV.ipynb)
Which load:
[notebooks/signals_2023-12-01T10_09_16.csv](notebooks/signals_2023-12-01T10_09_16.csv)
[notebooks/schemas_from_views_2023-12-01T10_09_16.csv](notebooks/schemas_from_views_2023-12-01T10_09_16.csv)

other data sources (from Mark' email)
You are each working on one of a number of POC’s leveraging LLM/ML tech to process natural language mining queries and generate SIGNAL or widget configurations for PI.
Let’s get together and discuss how we can support each other with a central training dataset.
Fareed has offered to support us with training data coming from various value accelerator resources and has shared some links to various github respositories that contain:

SIGNAL metrics developed for value accelerators.
Each query is associated with a human-readable explanation
20 metrics per accelerator
Example: https://github.com/signavio/pex-metric-library/blob/main/metric_definitions/invoice_to_cash.json
Documentation for the same metrics in .md format
Example: https://github.com/signavio/pi-connector-documentation/blob/master/Metrics%20library/Invoice-to-Cash/Invoice-to-Cash.md
Dashboards developed for accelerators.
JSON files containing all the formulas and configs for each widget, including the description and title, which are in human readable form.
Example: https://github.tools.sap/SAP-Signavio-PIA/pia-dashboards
</details>


# Preliminary results



## I. Text2Signal study: How 156-page document can be injected into the LLM prompt?

In this study we use the **human-created** textual description of the Signal queries as an input to LLM to reconstruct SIGNAL queries.
**Goal** of this study we measure how good a LLM understands SIGNAL language injected as a 156-pages long documentation text string into the prompt.

We validate the correctness of the LLM-generated SIGNAL queries by direct substitution to the Signavio UI.
Two criteria are applied:

- correctness of the formal SIGNAL syntax
- correctness of the results obtained by running the LLM-generated query against real data

<details>
<summary><i>Input length limitations, gpt4-32k prices </i></summary>

- *PDF->OCR->TXT* 156 pages gpt4 token length problem: 52801, while maximal length 32768 tokens
- *HTML->TXT* docs is about ~29k gpt4 tokens:

```js
  {'total_tokens': '29245',
  'total_cost_$': '0.043951000000000004',
  'prompt_tokens': '29078',
  'completion_tokens': '167',
  'latency': 34.23758792877197}
```

</details>

<details>
<summary><i>input: prompt to generate Signal query from human description with injected 156-pages Signal documentation </i></summary>

This is one of the prompts used where some additional requirements were added: [SignavioText2Signal/prompts/defaultGeneric.yaml](prompts/defaultGeneric.yaml)

```yaml
prompt: |
    You are SIGNAL assistant, a part of SAP Signavio's Process Intelligence Suite.
    SIGNAL stands for Signavio Analytics Language. Your goal is to help users craft SIGNAL
    queries and understand the SIGNAL language better. Here is a dump of
    the technical documentation:

    {tech_doc}


    Task to solve:

    Use the above SIGNAL documentation to construct SIGNAL expression querying
    {signal_description}

    Follow these instructions literally:
    - Use strict SIGNAL syntax.
    - When referencing column in GROUP BY clauses, please use column index.
    - Instead of the HAVING clause, use a subquery with a WHERE clause.
    - Explain how the query was constructed.

output_parser: pydantic_custom

```

</details>

<details>
<summary><i>input/output/gt: query N.7 </i></summary>

Example of the data used taken from


- **{tech_doc}** [data/text2signal-1-156.txt](data/text2signal-1-156.txt)

- **{signal_description}** [data/GT/text4signal_007.json](data/GT/text4signal_007.json):

```text
"Count of cases where invoice payment has not been received and customer risk class has not been assessed. We calculate this by counting the cases exhibiting specific behaviours, specifically where the case level attribute 'Risk Class' = 'Not Evaluated' and the event 'Create Invoice' is not followed by the event 'Receive Payment' or 'Cancel Invoice'. A behaviour is an expression evaluating case or event level attributes and can be used in a pattern matching expression."

```

Complete content of [data/GT/text4signal_007.json](data/GT/text4signal_007.json)

```yaml
{
    "category": "Pattern Matching / Deviations",
    "name": "Count of unpaid invoices with no risk assessment (Standard Process Deviation)",
    "details": "Count of cases where invoice payment has not been received and customer risk class has not been assessed. We calculate this by counting the cases exhibiting specific behaviours, specifically where the case level attribute 'Risk Class' = 'Not Evaluated' and the event 'Create Invoice' is not followed by the event 'Receive Payment' or 'Cancel Invoice'. A behaviour is an expression evaluating case or event level attributes and can be used in a pattern matching expression.",
    "query": "SELECT COUNT(1) AS \"Case Count\"\n          FROM THIS_PROCESS\n          WHERE \"Risk Class\" = 'Not evaluated'\n          AND\n          BEHAVIOUR\n                   (event_name = 'Create Invoice') AS create_invoice,\n                   (\"Risk Class\" = 'Not evaluated') AS no_risk_evaluation\n          MATCHES (no_risk_evaluation ~> create_invoice)\n          AND\n          NOT event_name MATCHES ('Create Invoice' ~> 'Receive Payment')\n          AND\n          NOT event_name MATCHES ('Create Invoice' ~> 'Cancel Invoice')\n",
    "expected_result": "\nCase Count\n102\n",
    "comment": "Numeric result",
    "update 1 18 Oct": "'Not evaluated' instead 'Not Evaluated'",
    "update 2 18 Oct": "Try to teach LLM that there are other events in between creation and event invoice payed"
}
```

</details>

### Text2Signal first evaluation results (manual)

With manual substitution of the LLM-generated queries  

```js
RUN:
SignavioText2Signal$ python eval.py  
OUTPUT:
predictions/
└── predictions-FromS2T-2023-12-01Tgpt-4-32k
    ├── text4signal_001.json
    ├── text4signal_002.json
    ├── text4signal_003.json
    ....
```

| LLM \ Query    |  Query 1  |  Query 2  |  Query 3  |  Query 4  |  Query 5  |  Query 6  |  Query 7  |  Query 8  |   Query 9  | Date |
| --------------: | ---------:|--------: | ---------:|--------: | ---------:|--------: | ---------:|--------: | ---------:| :-------: |
|  **gpt4-32k** (tuned)||||||||||  20-10-2023 |
|    Syntax    | ok | ok | ok | ok | ok | ok| ok  | ok | ok   |
|    Result    | ok | ok | ok | ok | ok | ok| Err | Err| ?    |
| **anthropic-claude-v1-100k** ||||||||||  20-10-2023 |
|    Syntax    |  ? | ok |  Err | Err | Err |  ok |  ok | Err | Err |
|    Result    | ok | ok |  Err | Err | Err | Err | Err | Err | Err |
| **anthropic-claude-v2-100k** ||||||||||  23-10-2023 |
|    Syntax    |  ok | ok |  ? | Err | Err | ok  | Err | Err | Err |
|    Result    |  ok | ok | ok | Err | Err | Err | Err | Err | Err |

**?-** means that LLM predicted syntax error can be easily corrected.

## II. Signal2Text -> Text2Signal evaluation results (manual substitution)

In this study we use LLM twice.
In the first phase we provide a valid SIGNAL query and ask LLM to summarize it.

<details>
<summary><i>input: prompt to summarize the Signal query with injected 156-pages Signal documentation </i></summary>

This is one of the prompts used where some additional requirements were added: [SignavioText2Signal/prompts/default_reciprocal.yaml](prompts/default_reciprocal.yaml)

```yaml
prompt: |
    You are SIGNAL assistant, a part of SAP Signavio's Process Intelligence Suite.
    SIGNAL stands for Signavio Analytics Language. Your goal is to help users craft SIGNAL
    queries and understand the SIGNAL language better. Here is a dump of
    the technical documentation:

    {tech_doc}


    Task to solve:

    Summarize the given SIGNAL query as a human-readable description.
    SIGNAL query:
    {signal}

    Form the output in JSON format:
    {{
    "case": {case},
    "query": {signal},
    "signal_description": text of the Signal summary should be here.
    }}

output_parser: pydantic_custom
```

- **{signal}**:

```JS
SELECT COUNT(1) AS "Case Count"
          FROM THIS_PROCESS
          WHERE "Risk Class" = 'Not evaluated'
          AND
          BEHAVIOUR
                   (event_name = 'Create Invoice') AS create_invoice,
                   ("Risk Class" = 'Not evaluated') AS no_risk_evaluation
          MATCHES (no_risk_evaluation ~> create_invoice)
          AND
          NOT event_name MATCHES ('Create Invoice' ~> 'Receive Payment')
          AND
          NOT event_name MATCHES ('Create Invoice' ~> 'Cancel Invoice')
```

</details>

In the second phase we use the LLM-generated summarizations/descriptions to generate SIGNAL query.

| LLM \ Query    |  Query 1  |  Query 2  |  Query 3  |  Query 4  |  Query 5  |  Query 6  |  Query 7  |  Query 8  |   Query 9  | Date |
| --------------: | ---------:|--------: | ---------:|--------: | ---------:|--------: | ---------:|--------: | ---------:| :-------: |
|  **gpt4-32k** p0||||||||||  23-10-2023 16:00 |
|    Syntax    |  ok | ok | ok | ?  | Err | ?  | Err | ?   | Err |
|    Result    |  ok | ok | ok | ok | Err | ok | Err | Err | Err |
|  **gpt4-32k** p1||||||||||  24-10-2023 8:00 |
|    Syntax    |  ok | Err |  ?  | ?  |  Err | ?  | Err | Err | Err |
|    Result    |  ok | Err |  ok | ok |  Err | ok | Err | Err | Err |
|  **gpt4-32k** pr3||||||||||  24-10-2023 13:00 |
|    Syntax    |  ok | ok | Err | Err | Err | ?  | ok  | ok  | ok  |
|    Result    |  ok | ok | Err | Err | Err | ok | Err | Err | ?Err|


## III. Text2Signal with Retrieval-Augmented Generation approach

In this approach we split the Signal documentation file into pieces of predefined length, vectorize, store and load these pieces into Chroma DB.

```py
    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

Then we use the vectorized Signal description to find `db.similarity_search(query,k=4)` K most semantically closest pieces of the Signal documentation.
I.e., we retrieve the top-k most relevant chunks by extracting the closest embedded chunks to our embedded query (Signal description).

Original text of these pieces are injected into the prompt by adding them as a string to the **{tech_rag}** clause.


<details>
<summary><i> Limitations: gpt-4 supports 8k tokens in the content window - so maximal number of chunks (1000) is *K=26* </i></summary>

For comparison prompt with K=4 pieces of documentation each 1000 tokens long is shown:

```js
K=26 # Max for 1000-long chunks
{
Total Cost (USD): $0.011414500000000001,
'datetime': '2023-11-02T14_26_27',
'latency': 20.634613037109375,
"completion_tokens": 425,
"prompt_tokens": 7043,
"total_tokens": 7468
}

K=5
 {
Total Cost (USD): $0.003647,
'datetime': '2023-11-02T11_51_00',
'latency': 19.12104296684265,
"completion_tokens": 433,
"prompt_tokens": 1854,
"total_tokens": 2287
}
```
</details>


<details>
<summary><i>input: RAG prompt to generate Signal query with injected K pieces from Signal documentation </i></summary>

The respective RAG prompt reads [SignavioText2Signal/prompts/defaultRAG.yaml](prompts/defaultRAG.yaml):

```yaml
prompt: |
    You are SIGNAL assistant, a part of SAP Signavio's Process Intelligence Suite.
    SIGNAL stands for Signavio Analytics Language. Your goal is to help users craft SIGNAL
    queries and understand the SIGNAL language better.

    Task to solve:

    {signal_description}

    Follow these instructions literally:
    - Use the dump of the SIGNAL technical documentation below to create task query with strict SIGNAL syntax.
    - When referencing column in GROUP BY clauses, please use column index.
    - Instead of the HAVING clause, use WHERE clause.
    - Explain how the query was constructed.

    Here is a dump of the SIGNAL technical documentation:

    {tech_rag}


output_parser: pydantic_custom

```

</details>


| LLM \ Query    |  Query 1  |  Query 2  |  Query 3  |  Query 4  |  Query 5  |  Query 6  |  Query 7  |  Query 8  |   Query 9  | Date |
| --------------: | ---------:|--------: | ---------:|--------: | ---------:|--------: | ---------:|--------: | ---------:| :-------: |
|  **gpt-4 K=26** ||||||||||  03-11-2023 8:00 |
|    Syntax    |  ok | ok | Err | Err | Err | ?  | ok  | ok  | Err |
|    Result    |  ok | ok | Err | Err | Err | ok | Err | Err | Err |
|  **anthropic-claude-v2 K=26** ||||||||||  03-11-2023 10:00 |
|    Syntax    |  ok |  ? | Err | Err | Err | Err | ok  | ?  | ok |
|    Result    |  ok | ok | Err | Err | Err | Err | Err | Err | Err |


# Discussion

## TEXT2SIGNAL

<details>
<summary><i>Query 4. Use-case: Variants. Example 2. Top X Variants by Percentile
 </i></summary>

Prompt original (as in PoC) version:

```JS
{"case":"Top X Variants by Percentile",
"signal_description":"Your goal is to construct SIGNAL query with maximum cycle time in milliseconds of 90% of all cases within each variant. For the purposes of the demo, we limit the results to the top 5 results. A process variant is a unique sequence of recorded events in a process (sometimes referred to as a trace)."}
```

Result in **WRONG** SIGNAL format:

```sql
SELECT
    event_name AS "Variant",
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY (SELECT LAST(end_time) - FIRST(end_time))) AS "90% Cycle Time (ms)"
FROM THIS_PROCESS
GROUP BY event_name
ORDER BY 2 DESC
LIMIT 5
```

Improved prompt: To resolve the above problem one has to add instruction: `When referencing column in GROUP by closures Please use column index.`

```sql
SELECT
    event_name AS "Variant",
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY (SELECT LAST(end_time) - FIRST(end_time))) AS "90% Cycle Time (ms)"
FROM THIS_PROCESS
GROUP BY 1
ORDER BY 2 DESC
LIMIT 5
```

</details>

<details>
<summary><i>Query 6: Compliance Rate for Standard Invoice Processing </i></summary>


- *GT*  Query 6: Compliance Rate for Standard Invoice Processing

```js
{
    "category": "Compliance",
    "name": "Compliance Rate for Standard Invoice Processing",
    "details": "compliance rate for standard invoicing with payment received. Compliance rate represents the count of cases exhibiting specific patterns / behaviours / sequences of events, divided by the total number of cases. In this example we combine SIGNAL FILTER and MATCHES expressions count cases exhibiting a specific event pattern.",
    "query": "SELECT ( \n COUNT(1) FILTER (WHERE event_name MATCHES ('Create Invoice'~> 'Receive Payment')) / COUNT(1) \n ) * 100 AS 'Standard Invoicing Compliance Rate' \n FROM THIS_PROCESS"
}
```

Input prompt:

```YAML
prompt: |
    You are SIGNAL assistant, a part of SAP Signavio's Process Intelligence Suite.
    SIGNAL stands for Signavio Analytics Language.Your goal is to help users craft SIGNAL
    queries and understand the SIGNAL language better. Here is a dump of
    the technical documentation:

    {{tech_doc}}

    Your goal is to construct SIGNAL query for the above data which would show the compliance
    rate for standard invoicing with payment received. The respective event names are 'Create Invoice' and
    'Receive Payment'. Compliance rate represents the count of cases exhibiting
    specific patterns / behaviours / sequences of events, divided by the total number of cases.
    Combine SIGNAL FILTER and MATCHES expressions count cases exhibiting a specific event pattern.

output_parser: pydantic_custom
```

#### gpt4-32k output

```js
"To calculate the compliance rate for standard invoicing with payment received, you can use the MATCHES expression to identify the specific event pattern and the FILTER expression to count the cases exhibiting this pattern. Here is an example of how you can construct this query:\n\n

\nSELECT \n    COUNT(case_id) FILTER (WHERE event_name MATCHES ('Create Invoice' ~> 'Receive Payment')) / COUNT(case_id) * 100 AS \"Compliance Rate\"\nFROM THIS_PROCESS\n

In this query, the MATCHES expression is used to identify the specific event pattern 'Create Invoice' followed by 'Receive Payment'. The FILTER expression is then used to count the cases that exhibit this pattern. This count is divided by the total number of cases to calculate the compliance rate. The result is multiplied by 100 to convert it into a percentage."
```

Relevant part of the LLM answer:

```JS
SELECT
    COUNT(case_id) FILTER (WHERE event_name MATCHES ('Create Invoice' ~> 'Receive Payment')) / COUNT(case_id) * 100 AS "Compliance Rate"
FROM THIS_PROCESS
```

</details>
