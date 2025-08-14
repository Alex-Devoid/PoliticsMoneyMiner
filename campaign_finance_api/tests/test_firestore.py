import os
import pytest
from fastapi.testclient import TestClient
from cloud_function.main import app

# Set environment variable for testing (Firestore emulator)
os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8180"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

client = TestClient(app)

def test_add_document():
    response = client.post("/documents/test_collection", json={"field": "value"})
    assert response.status_code == 200
    assert response.json()["success"] is True

def test_get_document():
    # Add a test document
    client.post("/documents/test_collection", json={"field": "value"})

    # Retrieve the document
    response = client.get("/documents/test_collection/<doc_id>")
    assert response.status_code == 200
    assert "field" in response.json()

def test_delete_document():
    # Add a test document
    response = client.post("/documents/test_collection", json={"field": "value"})
    doc_id = response.json()["doc_id"]

    # Delete the document
    response = client.delete(f"/documents/test_collection/{doc_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True
