"""Signal SQL parser."""
import json
import re

from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialect
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class Signavio(Dialect):
    """Signal SQL dialect."""

    class Tokenizer(Tokenizer):
        """Signal SQL tokenizer."""

        QUOTES = ["'", '"']
        IDENTIFIERS = ["`"]

        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "INT64": TokenType.BIGINT,
            "FLOAT64": TokenType.DOUBLE,
            "MATCHES": TokenType.LIKE,
            "~>": TokenType.ARROW,
            "BEHAVIOUR": TokenType.COMMAND,
            "FILL": TokenType.SELECT,
        }

    class Generator(Generator):
        """Signal SQL generator."""

        TRANSFORMS = {exp.Array: lambda self, e: f"[{self.expressions(e)}]"}

        TYPE_MAPPING = {
            exp.DataType.Type.TINYINT: "INT64",
            exp.DataType.Type.SMALLINT: "INT64",
            exp.DataType.Type.INT: "INT64",
            exp.DataType.Type.BIGINT: "INT64",
            exp.DataType.Type.DECIMAL: "NUMERIC",
            exp.DataType.Type.FLOAT: "FLOAT64",
            exp.DataType.Type.DOUBLE: "FLOAT64",
            exp.DataType.Type.BOOLEAN: "BOOL",
            exp.DataType.Type.TEXT: "STRING",
        }


def get_column_names_values(query, dialect=Signavio):
    """Parse SQL query and return column names and values."""
    query = query.replace("'", '"').replace('"', '"').replace("\n", " ").replace("\t", " ").replace("'", '"')
    try:
        out = {
            "parser_column_names": list({column.name for column in parse_one(query, dialect).find_all(exp.Column)}),
            "parser_values": list({column.name for column in parse_one(query, dialect).find_all(exp.Literal)}),
            "parser_error": "",
            "parser": "Signavio",
        }
    except Exception as e:
        print("SQL parsing error:", e)
        v = query.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ").replace(",", " ")
        out = {
            "parser_column_names": list({e for e in v.split(" ") if "_" in str(e)}),
            "parser_values": list(set(re.findall("'([^']*)'", query) + re.findall('"([^"]*)"', query))),
            "parser_error": e,
            "parser": "regexp",
        }
    return [
        json.dumps(out["parser_column_names"]),
        json.dumps(out["parser_values"]),
        out["parser_error"],
        out["parser"],
    ]
