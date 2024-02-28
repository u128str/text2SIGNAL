"""Utility functions for data processing."""
import json
from pathlib import Path

import yaml


def load_json(filename: str):
    """Load a JSON file."""
    with open(filename) as f:
        return json.load(f)


def save_json_to_file(data, file_path):
    """Save a JSON object to a file."""
    Path(file_path.parent).mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)


def sanitize_path_segment(segment):
    """Replace '/' with '_' in a path segment."""
    return segment.replace("/", "_")


def load_configurations(file_path):
    """Load a YAML file."""
    with open(file_path) as file:
        return yaml.safe_load(file)
