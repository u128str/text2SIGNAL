"""A module for exporting Signavio entities (Investigations and Dashboards) to the filesystem."""
import re
from pathlib import Path

import requests

from text2signal.authenticator import WORKSPACES, initialize_signavio_client


class SignavioExporter:
    """A class for exporting Signavio entities (Investigations and Dashboards) to the filesystem."""

    def __init__(self, signavio_client):
        """Initialize the signavio_client."""
        self.signavio_client = signavio_client

    def export_entity(self, entity_type, entity_id, target_dir=None, save_file=True):
        """Export an entity (Investigation or Dashboard) to the filesystem."""
        response = self.perform_request(entity_type, entity_id)
        if save_file:
            self._save_file_to_fs(response, target_dir, entity_type)

    def perform_request(self, entity_type, entity_id):
        """Construct the request for fetching the entity export and performs it."""
        endpoint_suffix_map = {
            "investigations": f"investigations/{entity_id}/export",
            "dashboards": f"dashboards/{entity_id}/export",
        }

        if entity_type not in endpoint_suffix_map:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        endpoint = (
            f"{self.signavio_client.pi.client_query.base_url}/g/api/pi-graphql/{endpoint_suffix_map[entity_type]}"
        )

        headers, cookies = self._prepare_auth()
        response = requests.get(endpoint, headers=headers, cookies=cookies, stream=True)
        response.raise_for_status()
        return response

    def _save_file_to_fs(self, response, target_dir, entity_type):
        """Save the response content to the filesystem."""
        file_path = self._construct_file_path(response, target_dir, entity_type)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return file_path

    def _construct_file_path(self, response, target_dir, entity_type):
        """Construct the file path for the exported entity."""
        filename = self._get_filename_from_cd(response.headers.get("content-disposition"))
        if not filename:
            raise Exception("Failed to extract filename from response headers")

        file_path = Path(target_dir) / entity_type / filename
        return file_path

    def _prepare_auth(self):
        """Prepare the authentication headers and cookies for the request."""
        headers = {"Accept": "*/*"}
        headers.update(self.signavio_client.pi.auth.headers)
        cookies = self.signavio_client.pi.auth.cookies
        return headers, cookies

    @staticmethod
    def _get_filename_from_cd(content_disposition):
        """Extract filename from the Content-Disposition header."""
        if content_disposition:
            filenames = re.findall('filename="([^"]+)"', content_disposition)
            return filenames[0] if filenames else None
        return None


def main():
    """Extract PI entity using the SignavioExporter."""
    tenant_id = WORKSPACES.get("Solutions Demo Workspace")
    signavio_client = initialize_signavio_client(tenant_id)
    exporter = SignavioExporter(signavio_client)
    entity_type = "dashboards"  # or 'investigations'
    entity_id = "01-cycle-time-dashboard-10"
    target_dir = "data/debug"
    exporter.export_entity(entity_type, entity_id, target_dir, save_file=False)


if __name__ == "__main__":
    main()
