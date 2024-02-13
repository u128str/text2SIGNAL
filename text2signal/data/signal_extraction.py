"""Extract signals from JSON files."""
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from string import Formatter, Template

import pandas as pd

from text2signal.data.utils import load_configurations, load_json, save_json_to_file


def populate_mapping_and_vars(metrics_jsons, mvars, key, var_mapping, metric_vars):
    """Populate mapping and metric variables."""
    if key in metrics_jsons:
        for var in metrics_jsons[key]:
            if var["name"] in mvars:
                var_mapping[var["name"]] = var.get("value", var.get("defaultValues", [{}])[0].get("defaultValue"))
                metric_vars[var["name"]] = var


def vars_substitution(mindex=0, metrics_jsons=None):
    """
    Substitute variables in a metric's signal fragment with their actual values.

    Args:
        mindex: Index of the metric in the metrics list.
        metrics_jsons: JSON object containing metrics and variables.

    Returns:
        Tuple containing:
        - The substituted signal fragment.
        - The original signal fragment.
        - A dictionary mapping variables to their substituted values.
        - The metric object at the provided index.
        - A dictionary of metric variables with their details.
    """
    if metrics_jsons is None:
        metrics_jsons = {}
    signal_fragment = metrics_jsons["metrics"][mindex]["signalFragment"]
    temp = Template(signal_fragment)

    mvars = [ele[1] for ele in Formatter().parse(signal_fragment) if ele[1]]

    var_mapping = {}
    metric_vars = {}

    populate_mapping_and_vars(metrics_jsons, mvars, "variables", var_mapping, metric_vars)
    populate_mapping_and_vars(metrics_jsons, mvars, "metricsVariables", var_mapping, metric_vars)
    populate_mapping_and_vars(metrics_jsons, mvars, "metricVariables", var_mapping, metric_vars)

    # Ensure all variables have a mapping
    for mv in mvars:
        if mv not in var_mapping:
            print(f"Warning: No substitution found for variable '{mv}'. Using placeholder.")
            var_mapping[mv] = mv

    substituted_signal = temp.safe_substitute(var_mapping)

    for mv in metric_vars:
        metric_vars[mv]["value"] = var_mapping[mv]

    return substituted_signal, signal_fragment, var_mapping, metrics_jsons["metrics"][mindex], metric_vars


def docs_from_metric_json(path, json_data=None, view="strview-1"):
    """Extract documents from a metric JSON file or object."""
    jdocs = []
    metrics = json_data if json_data else load_json(path)

    if "metrics" not in metrics:
        print(f"No 'metrics' found in provided JSON at {path} or in the given object.")
        return jdocs

    for i, _metric in enumerate(metrics["metrics"]):
        signal, original_signal, variable_map, metric_obj, metric_vars = vars_substitution(
            mindex=i, metrics_jsons=metrics
        )

        formatted_query = f"""
            SELECT
            {signal}
            FROM "{view}"
        """.strip()

        jdocs.append(
            {
                "name": metric_obj.get("name", f"Unnamed Metric {i}"),
                "query": formatted_query,
                "description": metric_obj.get("description", ""),
                "meta": "metric",
                "metric_vars": metric_vars,
                "view": view,
                "signalFragment": f"SELECT {original_signal} FROM THIS_PROCESS",
                "process": str(path),
            }
        )
    # return {str(path): jdocs}
    return jdocs


def extract_signals_from_json_section(path, json_section, view, meta, variables):
    """Extract signals from a JSON section."""

    def substitute_variables(signal_fragment, variables):
        """Replace placeholders with variable values in the signal fragment."""
        for name, details in variables.items():
            value = details.get("value", "")
            signal_fragment = signal_fragment.replace("${" + name + "}", str(value))
        return signal_fragment

    signals = []
    for widget in json_section.get("rootWidget", {}).get("children", []):
        for child in widget.get("children", []):
            if child.get("dataSource", {}).get("type") == "SIGNAL":
                signal_fragment = child["dataSource"]["query"]
                signal = substitute_variables(signal_fragment, variables)
                query = signal.replace('"THIS_PROCESS"', "THIS_PROCESS").replace("THIS_PROCESS", f'"{view}"')
                signals.append(
                    {
                        "name": child.get("name"),
                        "query": query,
                        "description": child.get("description", ""),
                        "meta": meta,
                        "metric_vars": variables,
                        "view": view,
                        "signalFragment": signal_fragment,
                        "process": str(path),
                    }
                )

    return signals


def docs_from_investigations_dashboard_json(path, json_data, source_keys, view):
    """Extract signals from investigations, dashboard, or metric JSONs."""
    if source_keys is None:
        source_keys = ["investigations", "dashboard", "metrics"]

    data = json_data if json_data is not None else load_json(path)

    variables = {var["name"]: var for var in data.get("metricsVariables", [])}
    signals = []
    # For dashboards and investigations, the structure within the JSON is similar
    for key in ["dashboard", "investigation"]:
        if key in source_keys:
            section = data.get(key, [])
            section_signals = extract_signals_from_json_section(path, section, view, key, variables)
            signals.extend(section_signals)
            # print(f"{path}{section} Signals: {section_signals}")

    # Metrics are structured differently and need separate handling if included in source_keys
    if "metrics" in source_keys and "metrics" in data:
        metrics_signals = docs_from_metric_json(path, json_data=data, view=view)
        signals.extend(metrics_signals)

    # return {str(path): signals}
    return signals


@dataclass
class SignalExtractor:
    """Class for the signal queries extraction."""

    workspace_name: str
    view_id: str
    base_path: Path

    def process_jsons(self):
        """Collect signals from the base path."""
        all_signals = []

        if self.base_path.is_file():
            all_signals += self.from_all_json_to_all_extracted_signals(load_json(self.base_path))
        elif self.base_path.is_dir():
            for file_path in self.base_path.rglob("*.json"):
                print(f"Processing file: {file_path}")
                try:
                    signals = []
                    json_data = load_json(file_path)
                    if "investigation" in json_data:
                        signals += docs_from_investigations_dashboard_json(
                            file_path, json_data, ["investigation", "metrics"], self.view_id
                        )
                    if "dashboard" in json_data:
                        signals += docs_from_investigations_dashboard_json(
                            file_path, json_data, ["dashboard", "metrics"], self.view_id
                        )
                    if "metrics" in json_data and "investigation" not in json_data and "dashboard" not in json_data:
                        signals += docs_from_metric_json(file_path, json_data)

                    for signal in signals:
                        signal.update(process="", origin=str(file_path), workspace_name=self.workspace_name)
                    all_signals.extend(signals)
                    print(f"{len(signals)}")

                except json.JSONDecodeError as e:
                    print(f"Invalid JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        else:
            print(f"Path {self.base_path} is neither a file nor a directory")

        print(f"Extracted {len(all_signals)} signals from {self.base_path}")
        return all_signals

    def extract_signals(self, process_name, json_files, extraction_func, source_keys=None):
        """Extract signals from JSON files using a given extraction function.

        Args:
            json_files: List of JSON files to process.
            extraction_func: Function to call for extracting signals.
            source_keys: List of source keys for extraction.

        Returns:
            A list of extracted signals.
        """
        signals = []
        for json_file_info in json_files:
            json_file_path = next(iter(json_file_info))
            json_data, view_id = json_file_info[json_file_path]["json"], json_file_info[json_file_path]["view_id"]

            if source_keys:
                extracted_signals = extraction_func(json_file_path, json_data, source_keys, view=view_id)
            else:
                extracted_signals = extraction_func(json_file_path, json_data, view=view_id)

            for signal in extracted_signals:
                signal.update(process=process_name, origin=str(json_file_path), workspace_name=self.workspace_name)

            signals.extend(extracted_signals)
            print(f"{source_keys}: {json_file_path} : {len(extracted_signals)}")

        return signals

    def from_all_json_to_all_extracted_signals(self, all_json):
        """Extract all signals from provided JSON data.

        Args:
            all_json: Dictionary containing JSON files.
            workspace_name: Name of the workspace being processed.

        Returns:
            A list of dictionaries containing signal information.
        """
        signals_all = []

        for process_name, content in all_json.items():
            signals_m = self.extract_signals(process_name, content["metrics_jsons"], docs_from_metric_json)
            signals_d = self.extract_signals(
                process_name,
                content["dashboards_jsons"],
                docs_from_investigations_dashboard_json,
                ["dashboard", "metrics"],
            )
            signals_i = self.extract_signals(
                process_name,
                content["investigations_jsons"],
                docs_from_investigations_dashboard_json,
                ["investigation", "metrics"],
            )

            signals = signals_m + signals_d + signals_i
            print(f"{process_name} subtotal Signals: {len(signals)}")
            signals_all.extend(signals)

        print(f"Total Signals: {len(signals_all)}")
        return signals_all


def main():
    """Extract signals from JSON files."""
    yaml_config = load_configurations("text2signal/configs/signal_extraction_config.yaml")
    print(f"Loaded configuration: {yaml_config.get('ouput_dir')}")
    configurations = yaml_config.get("configurations", [])
    all_signals = []

    for config in configurations:
        extractor = SignalExtractor(
            workspace_name=config["workspace"], view_id=config["view_id"], base_path=Path(config["path"])
        )
        signals = extractor.process_jsons()
        all_signals.extend(signals)

    print(f"Total signals extracted: {len(all_signals)}")
    # Save extracted signals
    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
    file_path = Path(f"{yaml_config.get('ouput_dir')}/all_extracted_signals_{date_str}.json")
    print(f"Dumping {len(all_signals)} extracted signals  to {file_path}")
    save_json_to_file(all_signals, file_path)

    df = pd.DataFrame(all_signals)
    signals_file_path = Path(f"{yaml_config.get('ouput_dir')}/signals_{date_str}.csv")
    print(f"Saving CSV file with Signals: {signals_file_path}")
    df.to_csv(signals_file_path)


if __name__ == "__main__":
    main()
