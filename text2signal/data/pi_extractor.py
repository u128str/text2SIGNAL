"""Module to fetch and save data from Signavio workspaces."""
import datetime
import os
from pathlib import Path

import requests

from text2signal.authenticator import initialize_signavio_client
from text2signal.data.pi_exporter import SignavioExporter
from text2signal.data.utils import load_configurations, sanitize_path_segment, save_json_to_file


class PIDataManager:
    """Manages the fetching and saving of data from Signavio workspaces."""

    def __init__(self, workspaces):
        """Initialize the Signavio clients for the given workspaces."""
        self.clients = {name: initialize_signavio_client(tenant_id) for name, tenant_id in workspaces.items()}

    @staticmethod
    def fetch_dashboards(fluffy, process_id):
        """Construct GraphQL query to fetch dashboards from PI."""
        dashboard_query = """
        query Dashboards($subjectId: ID!) {
            dashboards(subjectId: $subjectId) {
                access
                id
                name
                view {
                    name
                    id
                }
            }
        }
        """

        return fluffy.run(dashboard_query, values={"subjectId": process_id})

    @staticmethod
    def fetch_investigations(fluffy, process_id):
        """Construct GraphQL query to fetch investigation from PI."""
        investigations_query = """
        query Subject($subjectId: ID!) {
            subject(id: $subjectId) {
                id
                name
                investigationCount
                investigations {
                    id
                    name
                    view {
                        name
                        id
                    }
                }
            }
        }
        """
        return fluffy.run(investigations_query, values={"subjectId": process_id})

    @staticmethod
    def fetch_metrics(fluffy, process_id):
        """Construct GraphQL query to fetch metrics from PI."""
        metrics_query = """
        query metricsAndVariables($subjectId: ID!) {
            metrics(subjectId: $subjectId) {
                id
                name
                description
                signalFragment
                variables
                __typename
            }
            metricVariables(subjectId: $subjectId) {
                id
                name
                description
                value
                __typename
            }
        }
        """

        return fluffy.run(metrics_query, values={"subjectId": process_id})

    @staticmethod
    def fetch_columns(fluffy, view_id, process_id):
        """Construct GraphQL query to fetch all columns from PI."""
        columns_query = """
        query subjectView($id: ID!, $subjectId: ID!) {
            subjectView(id: $id, subjectId: $subjectId) {
                id
                name
                isDefaultView
                isAccessView
                columns {
                    name
                    isVisible
                    dataType
                    processVariableId
                    filter {
                        ... on SubjectViewColumnDateTimeFilter {
                            startDate
                            endDate
                            type
                            processVariableId
                            __typename
                        }
                        ... on SubjectViewColumnChoiceFilter {
                            includes
                            excludes
                            type
                            processVariableId
                            __typename
                        }
                        ... on SubjectViewColumnNumberFilter {
                            min
                            max
                            type
                            processVariableId
                            __typename
                        }
                        __typename
                    }
                    __typename
                }
                users {
                    id
                    role
                    entity {
                        ... on User {
                            id
                            firstName
                            lastName
                            name
                            email
                            type
                            __typename
                        }
                        ... on UserGroup {
                            id
                            name
                            type
                            __typename
                        }
                        ... on Everyone {
                            type
                            __typename
                        }
                        __typename
                    }
                    __typename
                }
                __typename
            }
        }
        """
        return fluffy.run(columns_query, values={"id": view_id, "subjectId": process_id})

    @staticmethod
    def process_entity(exporter, entity_list, entity_type, target_dir):
        """Process dashboards or investigations and returns JSON data."""
        entity_jsons = []

        for entity in entity_list:
            try:
                response = exporter.perform_request(entity_type, entity["id"])

                if response.status_code >= 200 and response.status_code < 300:
                    file_path = exporter._construct_file_path(response, target_dir, entity_type)
                    entity_json = response.json()
                    view_id = entity.get("view", {}).get("id", None)
                    entity_jsons.append({str(file_path): {"json": entity_json, "view_id": view_id}})
                else:
                    print(f"Request for entity {entity['id']} returned an error status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to perform request for entity {entity['id']}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for entity {entity['id']}: {e}")

        return entity_jsons

    def _initialize_for_workspace(self, workspace_name, root_dir):
        """Initialize and return necessary components for exporting data for a given workspace."""
        client = self.clients[workspace_name]
        exporter = SignavioExporter(client)
        target_dir = Path(root_dir) / sanitize_path_segment(workspace_name)
        fluffy = client.pi.fluffy
        return client, exporter, target_dir, fluffy

    def fetch_data(self, workspace_name, root_dir):
        """Fetch dashboard, investigation, and metric data for a given workspace."""
        client, exporter, target_dir, fluffy = self._initialize_for_workspace(workspace_name, root_dir)

        # Fetch list of processes
        list_processes = client.pi.subjects()

        all_json = {}
        for process in list_processes:
            process_id = process.id
            process_name = process.name
            new_target_dir = target_dir / sanitize_path_segment(process_name)

            dashboards = self.fetch_dashboards(fluffy, process_id)
            investigations = self.fetch_investigations(fluffy, process_id)
            metrics = self.fetch_metrics(fluffy, process_id)

            dashboards_jsons = self.process_entity(exporter, dashboards["dashboards"], "dashboards", new_target_dir)
            investigations_jsons = self.process_entity(
                exporter, investigations["subject"]["investigations"], "investigations", new_target_dir
            )

            dashboard_count = len(dashboards["dashboards"])
            investigation_count = len(investigations["subject"]["investigations"])
            metric_count = len(metrics["metrics"])

            views = {}
            view_ids = set()
            view_id = ""
            if dashboard_count > 0:
                view_ids.update(d["view"]["id"] for d in dashboards["dashboards"])
            if investigation_count > 0:
                view_ids.update(i["view"]["id"] for i in investigations["subject"]["investigations"] if "view" in i)
            for view_id in view_ids:
                columns = self.fetch_columns(fluffy, view_id, process.id)
                views[view_id] = columns

            all_json[process_name] = {
                "process_id": process_id,
                "dashboardCount": dashboard_count,
                "investigationCount": investigation_count,
                "metricCount": metric_count,
                "list_dashboards": dashboards,
                "dashboards_jsons": dashboards_jsons,
                "list_investigations": investigations,
                "investigations_jsons": investigations_jsons,
                "list_metrics": metrics,
                "metrics_jsons": [
                    {
                        str(Path(os.path.join(new_target_dir, "metrics", "metrics.json"))): {
                            "json": metrics,
                            "view_id": view_id,
                        }
                    }
                ],
                "views": views,
            }
        return all_json

    def save_all_jsons(self, root_dir, workspace_name, all_json):
        """Save dashboard, investigation, and metric data to JSON files."""
        client, exporter, target_dir, fluffy = self._initialize_for_workspace(workspace_name, root_dir)

        if not os.path.isdir(target_dir):
            for p, content in all_json.items():
                for dashboard in content["dashboards_jsons"]:
                    for file_path, details in dashboard.items():
                        save_json_to_file(details["json"], Path(file_path))
                for investigation in content["investigations_jsons"]:
                    for file_path, details in investigation.items():
                        save_json_to_file(details["json"], Path(file_path))
                new_target_dir = target_dir / sanitize_path_segment(p)
                views_file_path = Path(os.path.join(new_target_dir, "views", "views.json"))
                metrics_file_path = Path(os.path.join(new_target_dir, "metrics", "metrics.json"))
                save_json_to_file(content["views"], views_file_path)
                save_json_to_file(content["list_metrics"], metrics_file_path)

            # Dumping all JSON data to a single file
            date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
            filename_json_all = Path(target_dir).parent / f"all_json_{workspace_name}_{date_str}.json".replace(" ", "_")
            save_json_to_file(all_json, filename_json_all)
            print(f"Dumping all extracted Dashboards, Investigations, Metrics data to {filename_json_all}")
            return all_json, filename_json_all

        else:
            print(f"Directory {target_dir} already exists.")
            return all_json, None


def main():
    """Extract data from PI using the PIDataManager."""
    # manager = PIDataManager(WORKSPACES)
    # root_dir='data/temp'
    # for workspace_name in WORKSPACES:
    #     print(f"Processing workspace: {workspace_name}")
    #     all_json = manager.fetch_data(workspace_name, root_dir)
    #     manager.save_all_jsons(root_dir, workspace_name, all_json)

    config = load_configurations("text2signal/configs/workspaces.yaml")
    workspaces = config.get("workspaces", {})
    root_dir = config.get("root_dir", "data/temp")

    manager = PIDataManager(workspaces)

    for workspace_name in workspaces.keys():
        print(f"Processing workspace: {workspace_name}")
        all_json = manager.fetch_data(workspace_name, root_dir)
        manager.save_all_jsons(root_dir, workspace_name, all_json)


if __name__ == "__main__":
    main()
