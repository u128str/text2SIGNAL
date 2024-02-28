"""Validate signals and update them with views."""
import datetime
import json
from pathlib import Path

import pandas as pd

from text2signal.authenticator import WORKSPACES, initialize_signavio_client
from text2signal.data.signal_parser import Signavio, get_column_names_values


def extract_query_variables(df):
    """Extract the column names, values, errors and parser from the query."""
    df[["parser_column_names", "parser_values", "parser_error", "parser"]] = df.apply(
        lambda x: get_column_names_values(x["query"], dialect=Signavio), axis=1
    ).to_list()
    return df


def drop_duplicates(df, columns=None):
    """Drop duplicates based on name, description and signalFragment."""
    if columns is None:
        columns = ["name", "description", "signalFragment"]
    df.drop_duplicates(columns, keep="first", inplace=True)
    return df


def validate_signal(signavio_client, query):
    """Validate the signal query and return the result."""
    try:
        response = signavio_client.signal.query(query)
        data = response.data if hasattr(response, "data") else {}
        data_preview = json.dumps(data)[:100]
        data_length = len(data)

        return "ok", data_preview, data_length
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}", "", 0


def validate_and_update_queries_with_views(df, possible_views, signavio_client):
    """Update the queries with views and validate them."""

    def try_and_validate_query(row):
        original_query = row["query"]
        needs_replacement = 'FROM ""' in original_query or 'FROM FLATTEN("")' in original_query

        if needs_replacement:
            for view in possible_views:
                rewritten_query = original_query.replace('FROM ""', f'FROM "{view}"').replace(
                    'FROM FLATTEN("")', f'FROM FLATTEN("{view}")'
                )
                validation_result = validate_signal(signavio_client, rewritten_query)

                if (
                    validation_result[0] == "ok"
                    and validation_result[1] != "[[null]]"
                    and validation_result[1] != "[]"
                    and validation_result[2] > 0
                ):
                    return pd.Series(
                        [rewritten_query, view, *validation_result],
                        index=["query", "view", "APIvalidated", "validationDataResponse", "validationDataLength"],
                    )
            # If no view leads to a successful validation, return error information
            return pd.Series(
                [original_query, "", "Error: No valid view found", "", 0],
                index=["query", "view", "APIvalidated", "validationDataResponse", "validationDataLength"],
            )
        else:
            validation_result = validate_signal(signavio_client, original_query)
            return pd.Series(
                [original_query, row.get("view", None), *validation_result],
                index=["query", "view", "APIvalidated", "validationDataResponse", "validationDataLength"],
            )

    validation_columns = ["query", "view", "APIvalidated", "validationDataResponse", "validationDataLength"]
    df[validation_columns] = df.apply(try_and_validate_query, axis=1)

    return df


def main():
    """Validate signals."""
    auth_clients = {
        workspace_name: initialize_signavio_client(workspace_id) for workspace_name, workspace_id in WORKSPACES.items()
    }
    workspace_name = "Solutions Demo Workspace"
    signavio_client = auth_clients[workspace_name]

    filename_signals = "notebooks/data/temp/signals_2024-02-12T18_02_18.csv"
    df = pd.read_csv(filename_signals)
    print("Number of unfiltered Signals:", df.shape)

    # Process data
    df = extract_query_variables(df)
    df1 = drop_duplicates(df.copy())
    print("Number of unique Signals after duplicate cleaning:", df1.shape)

    print(df1[["name", "description", "signalFragment", "view"]].describe().loc[["count", "unique"]])

    # possible_views = df['view'].dropna().unique().tolist()
    # dfp = update_queries_with_views(dfp,possible_views, signavio_client)
    possible_views = ["defaultview-320"]
    dfp = validate_and_update_queries_with_views(df1.copy(), possible_views, signavio_client)

    valid_pdf = dfp[
        (dfp["APIvalidated"] == "ok")
        & (dfp["validationDataLength"] >= 1)
        & (dfp["validationDataResponse"] != "[[null]]")
    ]

    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
    filepath_validated = Path(f"data/temp/subset_validated_{valid_pdf.shape[0]}_signals_{date_str}.csv")
    filepath_validated.parent.mkdir(parents=True, exist_ok=True)
    valid_pdf.to_csv(filepath_validated)

    print(f"WRITE CSV file with Validated Signals: {filepath_validated} shape {valid_pdf.shape}")


if __name__ == "__main__":
    main()
