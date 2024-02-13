"""Evaluation of the reciprocal model for this use case."""
import datetime
import json
import os
import pathlib
import time
from typing import Any

from custom_parser_reciprocal import PydanticOutputParserLC

# from llm_commons.langchain.btp_llm import init_llm
from llm_commons.langchain.proxy import init_llm

# from langchain.prompts import PromptTemplate
# from llm_commons.langchain.prompt_registry import current_prompts
from llm_commons.prompt_registry import current_prompts
from llm_eval_toolkit.data import BaseEntry, UseCase
from llm_eval_toolkit.llm import ExtractionPredictor

# from llm_eval_toolkit.eval import iterate_through_dataset
from llm_eval_toolkit.utils.processing.dataset import iterate_through_dataset
from llm_eval_toolkit.utils.schema import field_descriptions_from_dict_multiple
from pydantic import BaseModel, Field

# Create the use case and register use case specific prompts
HERE = pathlib.Path(__file__).parent
USE_CASE = UseCase("signavio_text2signal")
USE_CASE.PromptRegistry.register_group(
    "extractor",
    # default=HERE / 'prompts' / 'default_reciprocal.yaml', # 0
    default=HERE / "prompts" / "default_reciprocal_work.yaml",  #  for gpt4 few shot
    # default=HERE / 'prompts' / 'default_completion.yaml',
    # openai_chat=HERE / 'prompts' / 'openai-chat.yaml'
)


#    {{"Query": "{query}"
#    }}
# Definition of the output structure
class Signal2Text(BaseModel):
    """Pydantic model for the output of the reciprocal model."""

    signal_description: str = Field(...)
    # query: str = Field(...)
    # explanation: str | None = ...
    # column_names: Optional[List] #[Optional[Dict[str,List[str]]]]]
    llm_output: str = None  # Optional[str]
    error_message: str = None  # Optional[str]
    llm_input: str = None  # Optional[str]


# Definition of the input data files
# subfolder='SignalDescriptions-2-3'
# subfolder_gt='GT-2-3'
# subfolder='SignalDescriptions'
# subfolder='GT-2-3'
# subfolder_gt='GT-2-3'
# subfolder='GT'
# subfolder_gt='GT'

# INPUT DIR is defined here
# inputfolder="training_data_10_with_validated_signals_2023-12-05T13_12_55"
# inputfolder="training_data_3_with_validated_signals_2023-12-06T11_08_27"
inputfolder = "training_data_1398_with_validated_signals_2023-12-07T11_13_29"  # old ones used ti train Mistral
inputfolder = "training_data_1542_with_validated_signals_2024-02-01T12_18_40"
inputfolder = "training_data_1542_with_validated_signals_2024-02-01T12_18_40_3_samples"
subfolder_gt = inputfolder
subfolder = inputfolder


@USE_CASE.register_input_file(name="input", subfolder=subfolder_gt, pattern="*.json")
def read_signal(path):
    """Read the signal from a JSON file."""
    print("PATH", path)
    with pathlib.Path(path).open(encoding="utf8") as stream:
        return json.load(stream, strict=False)


@USE_CASE.register_input_file(name="ground_truth", subfolder=subfolder_gt, pattern="*.json")
def read_ground_truth(path):
    """Read the ground truth from a JSON file."""
    with pathlib.Path(path).open(encoding="utf8") as stream:
        return json.load(stream, strict=False)


@USE_CASE.register_input_file(name="tech_doc", subfolder=subfolder, pattern="*.json")
def read_tech_doc(path):
    """Read the technical document from a JSON file."""
    path_doc = pathlib.PurePath(HERE, "../data", "text2signal-fromhtmls.txt")
    with pathlib.Path(path_doc).open(encoding="utf8") as stream:
        return stream.read()


# Data to prompt inputs
@ExtractionPredictor.register_output_parser("pydantic_custom")
def create_pydantic_parser(config, model_specific_config, pydantic_object):
    """Create a PydanticOutputParserLC instance."""
    return PydanticOutputParserLC(pydantic_object=pydantic_object)


@USE_CASE.PromptRegistry.attach_group("extractor")
class LLMSignal2Text(ExtractionPredictor):
    """LLMSignal2Text class for the reciprocal model."""

    def generate_prompt_input(self, entry: BaseEntry) -> dict[str, str]:
        """Generate the prompt input."""
        return {
            "query": entry.input["query"],
            #'tech_doc': entry.tech_doc, #[0:3000]
            #'case': entry.case_name
        }


def evaluation_func(entry: BaseEntry, prediction: tuple[Any, Any]):
    """Compare prediction and ground_truth."""
    # print("evaluation_func0",prediction)
    # print("metadata evaluation_func1",prediction[1])
    # if prediction.dict()["Classification"] != entry.ground_truth["class"]:
    # print(f'Prediction:\n{prediction.dict()}')
    # print(f'Expectation:\n{entry.ground_truth}')
    # Add llm_input to the Prediction results prediction.result.llm_input

    inp = predictor.generate_prompt_input(entry)
    llm_input = predictor.chain.prompt.format(**inp)
    # llm_input=fill_descriptions(entry,llm,predictor)
    prediction.result.llm_input = llm_input

    print(f"Prediction:\n{prediction.result}")

    # print(dir(entry))
    # print(entry)
    # prediction[0].llm_input=entry.raw_entry
    # print('API error>>>>>>>>>>',prediction[1]) #['MyCustomHandler']['api_errors'])
    # if 'API error' in prediction[0].keys():
    #    prediction[1]["error"]=prediction[0]['API error'] # works we can modify it here


# Function to parse model specific field descriptions
@USE_CASE.PromptRegistry.attach_group("extractor")
def fill_descriptions(llm):
    """Fill the descriptions for the model specific fields."""
    descriptions = current_prompts(llm).get("field_descriptions", {})
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<LLM",current_prompts(llm))
    if descriptions is not None:
        field_descriptions_from_dict_multiple(Signal2Text, descriptions=descriptions)


if __name__ == "__main__":
    # time delta
    start = time.time()
    # register dataset
    base_folder = pathlib.PurePath(HERE, "../data")
    USE_CASE.register_dataset("testSignavioSignal2Text", base_folder=base_folder)

    config = {
        "deployment_id": "gpt-4",  # "anthropic-claude-v2",
        # "gpt-4", # 'num_parallel_calls': 1, 101m7.848s
        # "gpt-35-turbo", # 'num_parallel_calls': 10, 6m44.474s
        # 'llama2-70b-chat-hf', # done 48m with 'num_parallel_calls': 10
        # 'llama2-70b-chat-hf',  # do not forget to insert [INST] [/INST]
        # "gpt-4",
        # "gpt-35-turbo", # llama2-13b-chat-hf
        # "anthropic-claude-v1",
        # "anthropic-claude-v2",
        # "gpt-35-turbo",
        # "gpt-4",
        # "gpt-4-32k",
        #'anthropic-claude-v2-100k', #"gpt-4-32k",
        # #'anthropic-claude-v1-100k',
        "temperature": 0.0,
        "num_parallel_calls": 5,  # <- take care with rate limit
        "dataset": "testSignavioSignal2Text",
        "max_tokens": 1000,  # None 256 alephalpha 3000 , # =None #for completion in davinci
    }

    llm = init_llm(
        config["deployment_id"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    # print("STR llm",llm)
    predictor = LLMSignal2Text(
        llm, output_data_model=Signal2Text, config=config, verbose=True, use_openai_functions_api=False
    )

    # predictor.chain.prompt.format
    # predictor.chain.prompt.format(locale="hhkj",page_jd="<<DoC is here>"))
    # predictor.chain.prompt.format_prompt format_prompt

    fill_descriptions(llm)

    # OUTPUT folder
    from llm_eval_toolkit.utils.callbacks.store_predictions import JSONStorage

    date_str = datetime.datetime.now().strftime("%Y-%m-%dT")
    # base_folder / str('predictions-30-Nov-S2T-00-'+ str(config['deployment_id']))
    json_prediction_path = os.path.join(
        HERE, "predictions", str(f"reciprocal-S2T-{date_str}_") + str(config["deployment_id"])
    )
    # with inputfolder
    json_prediction_path = os.path.join(HERE, "data", str(f'{inputfolder}_{config["deployment_id"]}'))
    print("Predictions:", json_prediction_path)

    with JSONStorage(json_prediction_path, overwrite=True) as json_storage:
        a = iterate_through_dataset(
            use_case=USE_CASE,
            dataset=config["dataset"],
            predictor=predictor,
            num_parallel_calls=config["num_parallel_calls"],
            callbacks=[evaluation_func, json_storage]
            # callbacks=[evaluation_func,jsonl_storage, pickle_storage, json_storage]
        )

    # end=time.time()
    # print(f" LLM runs for {len(a)} samples took {end-start}. Per-sample: {(end-start)/len(a)} ")
    ### Analysis:
    for el in a:
        #    #gt, (pr, metadata) = el
        gt, pr = el
        #    print(">>>>>>>>>>>>>>>>> GT")
        #    print(gt)
        #    print(">>>>>>>>>>>>>>>>> PR")
        print(pr.metadata)
    #    #print(">>>>>>>>>>>>>>>>>Meta")
    #    #print(metadata.keys())
    #    #print(metadata)

    end = time.time()
    print(f" LLM runs for {len(a)} samples took {end-start}. Per-sample: {(end-start)/len(a)} ")
