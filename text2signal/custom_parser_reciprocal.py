import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

#from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser, OutputParserException


T = TypeVar("T", bound=BaseModel)

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return " LLM cannot isolate SIGNAL query"

class PydanticOutputParserLC(BaseOutputParser[T]):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[T]
    """The pydantic model to parse."""

    PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""
    PYDANTIC_FORMAT_INSTRUCTIONS = ""

    def parse(self, text: str) -> T:
        #print("PArser:",text)
        try:
            # Greedy search for 1st json candidate.
            match = re.search(
                r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str, strict=False)
            json_object["llm_output"]=str(text)
            json_object["llm_input"]=""
            print("Parser:",json_object)
            return self.pydantic_object.model_validate(json_object)
        

        except (json.JSONDecodeError, ValidationError) as e:
            print("PArser ERROR:",text)
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion. Got: {e}"
            #err_json={"query": str(msg), "signal_description": "LLM Parser Error", "llm_output": text, "explanation": text }
            err_json={ "error_message": str(msg), "signal_description": "LLM Parser Error", "llm_output": str(text) }
            #raise OutputParserException(msg, llm_output=text)
            return self.pydantic_object.model_validate(err_json)

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.model_json_schema()
        print("STR custom get_format_instructions", schema)
        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return self.PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "pydantic"
