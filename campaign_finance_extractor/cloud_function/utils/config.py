# Filename: cloud_function/config.py

import os
from google.cloud import secretmanager
import google.auth

def get_project_id():
    """
    Retrieve the Google Cloud project ID in which the function is running.
    """
    _, project_id = google.auth.default()
    return project_id

def access_secret_version(secret_id, version_id="latest"):
    """
    Access the secret version from Google Secret Manager.
    Args:
        secret_id (str): The ID of the secret to access.
        version_id (str): The version of the secret to access. Defaults to "latest".
    Returns:
        str: The secret payload.
    """
    client = secretmanager.SecretManagerServiceClient()
    project_id = get_project_id()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version
    response = client.access_secret_version(name=name)

    # Return the secret payload
    return response.payload.data.decode("UTF-8")
