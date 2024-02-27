"""Authentication logic."""
from signavio import CredentialAuthentication, Signavio

from text2signal.configs.config import env

WORKSPACES = {
    "Process AI": "b0f07deabd3140aea5344baa686e0d84",
    "Solutions Demo Workspace": "ccf98b692d1d4e0aa895df3aab8cd905",
}


def initialize_signavio_client(tenant_id):
    """Initialize a Signavio client with the provided tenant ID."""
    # Mapping of CloudOS names to their base URLs
    # CLOUDOS_URLS = {
    #     "production": "https://editor.signavio.com",
    #     "staging": "https://staging.signavio.com",
    # }

    user_name = env.get("MY_SIGNAVIO_NAME")
    password = env.get("MY_SIGNAVIO_PASSWORD")

    auth = CredentialAuthentication(username=user_name, password=password, tenant_id=tenant_id, cloud_os="production")
    signavio_client = Signavio(auth)
    return signavio_client


def main():
    """List processes in a given workspace."""
    auth_clients = {}

    for workspace_name, workspace_id in WORKSPACES.items():
        auth_clients[workspace_name] = initialize_signavio_client(workspace_id)

    workspace_name = "Process AI"  # or "Solutions Demo Workspace"
    signavio_client = auth_clients[workspace_name]

    pi_subjects = signavio_client.pi.subjects()
    print(pi_subjects)


if __name__ == "__main__":
    main()
