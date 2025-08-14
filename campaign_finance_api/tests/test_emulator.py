import os
import pytest
from google.cloud import firestore

# Set Firestore Emulator environment variables for testing
os.environ["FIRESTORE_EMULATOR_HOST"] = "firestore-emulator:8180"  # Update to match Docker setup
os.environ["GOOGLE_CLOUD_PROJECT"] = "content-audit-333003"

@pytest.fixture
def firestore_client():
    """
    Fixture to provide a Firestore client connected to the emulator.
    """
    return firestore.Client()

def test_firestore_write_and_read(firestore_client):
    """
    Test writing a document to the Firestore emulator and reading it back.
    """
    collection_name = "scraped_documents"
    document_id = "test_doc"
    test_data = {"name": "Test Document", "value": 42}

    # Write a document
    doc_ref = firestore_client.collection(collection_name).document(document_id)
    doc_ref.set(test_data)

    # Read the document back
    fetched_doc = doc_ref.get()
    assert fetched_doc.exists, "Document does not exist in Firestore."
    fetched_data = fetched_doc.to_dict()

    # Assert that the fetched data matches the input data
    assert fetched_data == test_data, f"Fetched data {fetched_data} does not match expected {test_data}."
