"""Fetch schema information for all processes from PI, including view information from all_json, and prepares a DataFrame."""
import pandas as pd

from text2signal.authenticator import WORKSPACES, initialize_signavio_client
from text2signal.data.utils import load_json


def fetch_and_prepare_schema_data(signavio_client, all_json):
    """
    Fetch schema information for all processes from PI, including view information from all_json, and prepares a DataFrame.

    Args:
        signavio_client: The Signavio client configured to access the API.
        all_json: The loaded JSON data containing additional information such as views.
    """
    schema_data = []

    for process in signavio_client.pi.subjects():
        print(f"Fetching schema for process: {process.id}, {process.name}")
        schema = signavio_client.signal.schema(process.id)

        # Extract 'view' for a given process
        p_info = all_json.get(process.name, {})
        if "views" in p_info:
            view = list(p_info["views"].keys())[0]
            print(f"Found views {list(p_info['views'].keys())} for process: {process.name}")

        schema_data = [
            {
                "column_name": field.column_name,
                "name": field.column_display_name,
                "column_role": field.column_role.name,
                "dataType": field.data_type.name,
                "view": view,
                "process": process.name,
                "process_id": process.id,
            }
            for field in schema.fields
        ]

    df = pd.DataFrame(schema_data)
    return df


def main():
    """Fetch schema information for all processes from PI, including view information from all_json, and prepares a DataFrame."""
    tenant_id = WORKSPACES.get("Solutions Demo Workspace")
    signavio_client = initialize_signavio_client(tenant_id)

    json_filepath = "data/From_API/all_json_Solutions_Demo_Workspace_2024-02-01T08_17_38.json"
    all_json = load_json(json_filepath)

    df_schema = fetch_and_prepare_schema_data(signavio_client, all_json)
    print(df_schema)

    # df_schema.to_csv('data/debug/signavio_schema_info.csv', index=False)


if __name__ == "__main__":
    main()
